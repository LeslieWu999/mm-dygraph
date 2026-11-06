# MM-DyGraph

This is the official implementation of NeurIPS 2026 paper *MM-DyGraph: A Dataset, Benchmark, and Model for Multimodal Dynamic Graphs*, including the baselines evaluation, the proposed TST-MF model, and the scripts that built the datasets.

Three tasks are evaluated on temporal graphs whose nodes and edges carry text and image features: link prediction, node retrieval and edge classification, in the transductive (*tr.*) and, for the first two, the inductive (*in.*) setting.

## Installation

Tested with Python 3.10.

```bash
pip install torch==2.8.0  # choose the build matching your CUDA version
pip install -r requirements.txt
```

## Data

The datasets are available on [Hugging Face](https://huggingface.co/datasets/LeslieWu999/MM-DyGraph). They are derived from GDELT, Google Local, Amazon Reviews 2023 and the Yelp Open Dataset.

Each one is a directory under a common root:

```
<data_root>/<dataset>/
├── edge_list.csv     # columns: u, r, i, ts, label
└── npy/
    ├── e_feat_cat.npy    # node features: text and image embeddings, concatenated
    ├── r_feat_cat.npy    # edge features: text and image embeddings, concatenated
    ├── e_feat_ran.npy    # random node features
    └── r_feat_ran.npy    # random edge features
```

In `edge_list.csv`, `u` and `i` are the source and destination node ids, which index the rows of the node feature tables, and `r` is the edge id, which indexes the rows of the edge feature tables. Node id 0 is reserved for padding.

Pass the root with `--data_root` or set `MMDG_DATA_ROOT`. To build a dataset in this format from your own raw data, see [data_construction/README.md](data_construction/README.md).

Full-data variants such as `gdelt_full`, `gmct_full` and `yelp_full` are ordinary dataset
directories with the same layout; select one by passing its directory name to `--dataset_name`.

## Usage

```bash
export MMDG_DATA_ROOT=/path/to/datasets

# Link prediction
python train_link_prediction.py --dataset_name GDELT --model_name DyGFormer --wrapper_name tst_mf --use_feature text+image

# Node retrieval, using the checkpoints of the link prediction run above
python evaluate_node_retrieval.py --dataset_name GDELT --model_name DyGFormer --wrapper_name tst_mf --use_feature text+image

# Edge classification
python train_edge_classification.py --dataset_name GDELT --model_name DyGFormer --wrapper_name tst_mf --use_feature text+image
```

The defaults match the settings of the paper: seeds 0, 1 and 2, at most 200 epochs, early stopping with a patience of 5 evaluations, and per-model hyperparameters from `utils/load_configs.py`. Use `--help` for all options. `--offload_features` keeps the feature tables in host memory; it does not change the results and saves over 10 GB of GPU memory per process on the larger datasets.

Use `--seeds 3,7,11` to run any explicit set of random seeds. This overrides `--num_runs`.

`scripts/run_benchmark.sh` runs the whole benchmark sequentially.

Results are written under `--output_dir`, by default the current directory:

```
saved_models/<model>/<dataset>/<run>/     # checkpoints
saved_results/<model>/<dataset>/<run>.json
logs/<model>/<dataset>/<run>/run.log
```

A run is identified by its task, wrapper, model, features and seed. Retrieval result names additionally
include the negative sampler, evaluation protocol and candidate count, so those variants do not
overwrite or merge with one another. Running other hyperparameter variants may replace existing
checkpoints and results, so use a different `--output_dir` for those.

In the result files, `test metrics` is the transductive setting and `new node test metrics` is the inductive one. `python -m scripts.collect_results` collects all of them into CSV files.

### Methods

A method combines a temporal backbone with a multimodal fusion wrapper.

| Method | `--model_name` | `--wrapper_name` |
|---|---|---|
| TGAT, CAWN, TCL, GraphMixer, DyGFormer, FreeDyG, TPNet | same as the method | `concat` |
| SALoM | `Liquid` | `concat` |
| JODIE, DyRep, TGN | same as the method | `concat` |
| Gated fusion | `DyGFormer` | `gated_fusion` |
| Modality attention | `DyGFormer` | `modality_attention` |
| UniGraph2 | `DyGFormer` | `unigraph2` |
| **TST-MF** | `DyGFormer` | `tst_mf` |
| TST-MF w/o target | `DyGFormer` | `tst_mf_no_target_guidance` |
| TST-MF w/o hier. | `DyGFormer` | `tst_mf_single_stage` |
| TST-MF w/o spatial | `DyGFormer` | `tst_mf_no_struct_agg` |
| TST-MF w/o historical | `DyGFormer` | `tst_mf_no_hist_tracker` |

The fusion wrappers `gated_fusion`, `modality_attention` and `unigraph2` work with any backbone. `--use_feature` joins modalities with `+`, out of `rand`, `text` and `image`.

Missing images are detected directly from the concatenated feature tables: dataset construction
duplicates the text embedding into the image half when an image is absent. The feature tokens are:

- `text+image_zero`: Zero Filling, replacing detected missing images with zeros;
- `text+image_mean`: Mean Filling, using the mean observed image embedding;
- `text+image_learned`: Learned Embedding, initialized to that mean.

Padding row 0 is excluded from missing-image detection and mean computation, and remains zero.

For recurrence-aware node retrieval, evaluate an existing link-prediction checkpoint with:

```bash
python evaluate_node_retrieval.py --dataset_name gdelt --model_name DyGFormer \
  --wrapper_name tst_mf --use_feature text+image \
  --negative_sample_strategy historical \
  --retrieval_evaluation_protocol recurrence_aware
```

This reports overall, repeated-edge and novel-edge H@1/H@3/H@10, MRR and mean rank. A directed
edge is repeated only if the same ordered pair occurred at a strictly earlier timestamp.
`--num_negative_candidates` controls the sampled candidate count (default: 100).

### Modality utility

```bash
python -m scripts.analyze_modality_utility --task lp --dataset_name GDELT --model_name DyGFormer --wrapper_name tst_mf
python -m scripts.summarize_modality_utility --datasets GDELT
```

## Repository layout

```
train_*.py, evaluate_node_retrieval.py   entry points
pipeline.py                              training and evaluation of a run
evaluation.py                            evaluation loops
models/                                  backbones (see THIRD_PARTY_NOTICES.md), fusion.py (TST-MF),
                                         fusion_baselines.py, adapters.py
utils/                                   data loading, arguments, model construction, metrics
scripts/                                 result collection and analyses
data_construction/                       scripts that turn raw data into this format
```

## Third-party code

We thank the authors of [DyGLib](https://github.com/yule-BUAA/DyGLib), our repository is adapted from their release. We also thank to the authors of [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs), [TGN](https://github.com/twitter-research/tgn), [CAWN](https://github.com/snap-stanford/CAW), [GraphMixer](https://github.com/CongWeilin/GraphMixer), [DyGFormer](https://github.com/yule-BUAA/DyGLib), [FreeDyG](https://github.com/Tianxzzz/FreeDyG), [TPNet](https://github.com/lxd99/TPNet), [SALoM](https://github.com/wave5418/SALoM), [JODIE](https://github.com/sizhewan/Predicting-Dynamic-Embedding-Trajectory), [DyRep](https://github.com/uoguelph-mlrg/LDG), and [UniGraph2](https://github.com/yf-he/UniGraph2) for making their project codes publicly available. The code from other projects under their own licenses, please refer to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

TODO
