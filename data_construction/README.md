# Dataset construction

The scripts in `common/` turn a raw multimodal temporal graph into the files that the training code reads.
The released datasets were built with these steps. The dataset-specific steps that collected and filtered
their raw data are not part of this repository. To build a dataset from your own data, write it in the
[input format](#input-format) below, then run the [steps](#steps).

```
data_construction/common/
  00_strip_image_paths.py       optional: reduce image paths to file names
  01_score_images.py            score each image against the text of its node or edge, with a VLM behind vLLM
  02_construct_graph.py         choose one image per node and edge, assign ids and time buckets, write the CSV files
  03_extract_clip_features.py   CLIP text and image features
  04_dataset_stats.py           node, edge, image and timestamp counts
```

## Requirements

- Python 3.10 or newer
- The training requirements, plus `pip install -r requirements.txt`
- A GPU server running [vLLM](https://github.com/vllm-project/vllm) with `Qwen/Qwen3-VL-8B-Instruct`, for image
  scoring. Any OpenAI-compatible chat completions endpoint that accepts base64 images works.
- `openai/clip-vit-base-patch16` from Hugging Face, for feature extraction. `--pretrained-model` also accepts a
  local directory, which must contain `model.safetensors`: a directory with only `pytorch_model.bin` does not load.

The commands below use these variables:

```bash
DC=/path/to/data_construction   # this directory
RAW=/path/to/raw                # holds nodes.jsonl and edges.jsonl
ROOT=/path/to/datasets          # final dataset root, later passed to training as --data_root or MMDG_DATA_ROOT
DATASET=my_dataset              # folder name under $ROOT
```

## Input format

```
$RAW/
  nodes.jsonl
  edges.jsonl
$ROOT/$DATASET/
  img_entity/      node images, in one flat directory
  img_relation/    edge images, in one flat directory
```

Both JSONL files are UTF-8 with one JSON object per line and no blank lines.

### `nodes.jsonl`

| Field | Type | Required | Content |
| --- | --- | --- | --- |
| `id` | string or integer | yes | Unique node key. Edges refer to it by value, so the type must match: `"7"` and `7` are different nodes. |
| `text` | string | no | Node text. |
| `image` | list | no | Candidate images of the node, see [Images](#images). |

### `edges.jsonl`

| Field | Type | Required | Content |
| --- | --- | --- | --- |
| `u_id` | string or integer | yes | Source node, an `id` of `nodes.jsonl`. |
| `v_id` | string or integer | yes | Destination node. Node retrieval ranks destination nodes. |
| `timestamp` | integer | yes | Time of the interaction. Any unit works if `--bucket-interval` uses the same unit; the released datasets use Unix milliseconds. |
| `label` | integer | no, default 0 | Class of the edge, for edge classification. Use consecutive integers that start at 0 or 1. |
| `text` | string | no | Edge text. |
| `image` | list | no | Candidate images of the edge, see [Images](#images). |

Other fields, such as `image_url`, are ignored. For example:

```json
{"id": "user:17", "text": "Joined in 2019, mostly reviews cafes."}
{"id": "business:42", "text": "Blue Door Cafe. Breakfast and coffee.", "image": ["business_42.jpg"]}
```

```json
{"u_id": "user:17", "v_id": "business:42", "timestamp": 1714521600000, "label": 5, "text": "Great pancakes.", "image": ["r_901_a.jpg", ["r_901_b.jpg", "photo of the menu"]]}
```

### Images

An `image` entry is a file name, or a `[file name, description]` pair whose description is added to the scoring
prompt. Only the base name of an entry is used, so `downloads/a.jpg` and `https://host/x/a.jpg` both refer to
`a.jpg`. The file is looked up in `img_entity/` for nodes and in `img_relation/` for edges, so file names must be
unique within each directory. Entries whose file is missing are skipped. Use JPEG, PNG or WebP files: the MIME type
sent to the scoring server is chosen from the extension.

`00_strip_image_paths.py` rewrites the entries of a JSONL file to their base names. The later steps give the same
result with or without it. If you run it, give its output to both `01_score_images.py` and `02_construct_graph.py`.

### How the records are used

- Edges can be in any order. They are sorted by `timestamp`, and edges with the same timestamp keep their file
  order.
- Only nodes that appear in an edge are kept. A node that an edge uses but `nodes.jsonl` lacks gets empty text and
  no image.
- A node or edge without text is not scored, so none of its images can be chosen. Empty text is encoded as the
  string `NULL`.

## Steps

Start the scoring server in a separate shell. Add `--tensor-parallel-size N` for multiple GPUs.

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

Then run:

```bash
# Score every image against the text of its edge, and of its node.
python $DC/common/01_score_images.py --input-file $RAW/edges.jsonl --image-dir $ROOT/$DATASET/img_relation \
    --output-file $RAW/edge_scores.jsonl \
    --api-url http://localhost:8000/v1/chat/completions --model Qwen/Qwen3-VL-8B-Instruct
python $DC/common/01_score_images.py --input-file $RAW/nodes.jsonl --image-dir $ROOT/$DATASET/img_entity \
    --output-file $RAW/node_scores.jsonl \
    --api-url http://localhost:8000/v1/chat/completions --model Qwen/Qwen3-VL-8B-Instruct

# Choose the images, assign ids and time buckets, and write the CSV files.
# --bucket-interval is the bucket width in the unit of timestamp (900000 ms = 15 minutes).
python $DC/common/02_construct_graph.py --input-edges $RAW/edges.jsonl --input-nodes $RAW/nodes.jsonl \
    --node-validation $RAW/node_scores.jsonl --edge-validation $RAW/edge_scores.jsonl \
    --output-dir $ROOT/$DATASET --bucket-interval 900000

# Extract CLIP features. Plain `python` also works, using a single process.
torchrun --nproc_per_node=4 $DC/common/03_extract_clip_features.py --dataset $DATASET --data-root $ROOT

# Print node, edge, image and timestamp counts.
python $DC/common/04_dataset_stats.py --data-root $ROOT --datasets $DATASET
```

If no node has an image, skip the node scoring and pass `--node-validation /dev/null`.

`02_construct_graph.py` builds the graph as follows:

- **Images.** An image is a candidate if its relevance score is at least 6.0. The candidate with the highest
  relevance is used, and ties are broken with a fixed seed. The consistency score is not used.
- **Ids.** Node and edge ids start at 1, in order of first appearance after sorting the edges by time. Id 0 is
  left for padding.
- **Time.** An edge's bucket is `floor(timestamp / interval) * interval`, so buckets are aligned to multiples of
  the interval, not to the first edge. `ts` is the index of the bucket among the buckets that contain at least one
  edge: empty buckets get no index, so a stretch of empty buckets becomes a step of 1.

## Final on-disk format

```
$ROOT/<dataset>/
  edge_list.csv        u, r, i, ts, label
  entity_attr.csv      i, img, text      one row per node
  relation_attr.csv    i, img, text      one row per edge
  img_entity/          node images
  img_relation/        edge images
  npy/
    e_feat_{cat,rep,ran}.npy    shape (num_nodes + 1, 1024), float32
    r_feat_{cat,rep,ran}.npy    shape (num_edges + 1, 1024), float32
```

- `u` and `i` in `edge_list.csv` are node ids and `r` is the edge id. The `i` column of the attribute files holds
  these same ids, and `img` is the chosen image file, empty when there is none.
- Row 0 of each feature array is all zeros and serves as padding.
- Each row is two 512-dimensional L2-normalized halves:
  - `cat` is text followed by image. A node or edge without a usable image uses its text features in the image
    half.
  - `rep` is text followed by text.
  - `ran` is random noise, normalized over the full 1024 dimensions rather than per half.
- Long texts are split into segments that fit CLIP's context length, and the segment features are averaged.

Training needs `edge_list.csv` and the `cat` and `ran` arrays. The loader applies two more transformations: it
divides `ts` by 15 for datasets whose name contains `gdelt` in any case, and it subtracts 1 from `label` when the
smallest label is not 0. Do not put `gdelt` in the name of another dataset.

## Reproducibility notes

- **Scoring is not deterministic.** Requests use temperature 0.2 without a seed. A pair that still fails after
  `--max-attempts` is scored -1, so its image is dropped. Server settings can cause this too: a text that exceeds
  the server's maximum length fails on every attempt.
- **Score ids are line numbers.** `01_score_images.py` identifies each pair by its line in the scored file.
  `02_construct_graph.py` must read that exact file, and a file must not be split into shards that are scored
  separately, because the ids would collide.
- **Missing score files.** `02_construct_graph.py` treats a missing score file as empty, which silently removes all
  images. Check the image ratios printed by `04_dataset_stats.py`.
- **Random features depend on the rank count.** `ran` is seeded per rank, so it changes with the number of processes.
- **Backslashes in text are doubled.** `02_construct_graph.py` writes the CSV files with `\` as the escape
  character, and `03_extract_clip_features.py` reads them without one, so each `\` becomes `\\` in the CSV text and
  in the text features. The released CSV files show the same doubling.
