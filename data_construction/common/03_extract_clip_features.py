"""Extract CLIP text and image features into the npy arrays consumed by the training code.

Writes {e,r}_feat_{cat,rep,ran}.npy under <output-root>/<dataset>/npy/. Runs as a single process
or under torchrun, in which case each rank handles a contiguous slice of the CSV.
"""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    # model
    PRETRAINED_MODEL: str = "openai/clip-vit-base-patch16"
    BATCH_SIZE: int = 2048          # CSV rows per batch, per rank
    TEXT_SEGMENT_BATCH_SIZE: int = 1024
    IMAGE_BATCH_SIZE: int = 256     # images need a much smaller batch than text
    PRECISION: str = "fp16"         # fp16 / bf16 / fp32
    SEED: int = 42

    # IO / merge
    SCAN_CHUNK_SIZE: int = 1_000_000
    MERGE_CHUNK_SIZE: int = 200_000
    KEEP_SHARDS: bool = False


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize distributed mode under torchrun; fall back to a single process otherwise."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def configure_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - rank={rank} - %(levelname)s - %(message)s",
        force=True,
    )


def barrier() -> None:
    if is_dist():
        dist.barrier()


def broadcast_int(value: Optional[int], device: torch.device, src: int = 0) -> int:
    backend_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([value if value is not None else -1], dtype=torch.long, device=backend_device)
    if is_dist():
        dist.broadcast(tensor, src=src)
    return int(tensor.item())


def seed_everything(seed: int, rank: int) -> None:
    # Each rank gets its own stream so the random features differ across ranks.
    seed = seed + rank * 10007
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def scan_csv_stats(csv_path: Path, chunksize: int) -> Tuple[int, int]:
    """Scan only the id column to get the row count and the largest id."""
    total_rows = 0
    max_i = 0
    for chunk in pd.read_csv(csv_path, usecols=["i"], dtype={"i": "int64"}, chunksize=chunksize):
        if len(chunk) == 0:
            continue
        total_rows += len(chunk)
        max_i = max(max_i, int(chunk["i"].max()))
    return total_rows, max_i


def get_csv_stats_distributed(csv_path: Path, config: Config, rank: int, device: torch.device) -> Tuple[int, int]:
    if rank == 0:
        logging.info(f"Scanning CSV stats: {csv_path}")
        total_rows, max_i = scan_csv_stats(csv_path, config.SCAN_CHUNK_SIZE)
        logging.info(f"CSV stats: rows={total_rows}, max_i={max_i}")
    else:
        total_rows, max_i = None, None

    total_rows = broadcast_int(total_rows, device=device, src=0)
    max_i = broadcast_int(max_i, device=device, src=0)
    return total_rows, max_i


def shard_bounds(total_rows: int, rank: int, world_size: int) -> Tuple[int, int]:
    per_rank = math.ceil(total_rows / world_size) if world_size > 0 else total_rows
    start = min(rank * per_rank, total_rows)
    end = min((rank + 1) * per_rank, total_rows)
    return start, end


def iter_csv_shard(
    csv_path: Path,
    start_row: int,
    num_rows: int,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """Read this rank's slice; skiprows keeps the header and skips data rows [1, start_row]."""
    if num_rows <= 0:
        return

    skiprows = range(1, start_row + 1) if start_row > 0 else None
    reader = pd.read_csv(
        csv_path,
        dtype={"i": "int64", "img": "string", "text": "string"},
        skiprows=skiprows,
        nrows=num_rows,
        chunksize=chunksize,
    )
    for chunk in reader:
        yield chunk


class MultimodalFeatureExtractor:
    def __init__(self, config: Config, device: torch.device, rank: int):
        self.config = config
        self.device = device
        self.rank = rank
        logging.info(f"Initializing CLIP on {self.device}")

        self.processor = CLIPProcessor.from_pretrained(config.PRETRAINED_MODEL, use_fast=True)

        # autocast only affects operators, so the weights are cast explicitly as well.
        if self.device.type == "cuda" and config.PRECISION == "fp16":
            model_dtype = torch.float16
        elif self.device.type == "cuda" and config.PRECISION == "bf16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32

        self.model = CLIPModel.from_pretrained(
            config.PRETRAINED_MODEL,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

        self.feature_dim = int(self.model.config.projection_dim)
        self.max_length = int(self.model.config.text_config.max_position_embeddings)
        self.tokenizer = self.processor.tokenizer
        self.cpu_generator = torch.Generator(device="cpu").manual_seed(config.SEED + rank * 10007)

    def _amp_context(self):
        if self.device.type != "cuda" or self.config.PRECISION == "fp32":
            return nullcontext()
        if self.config.PRECISION == "bf16":
            return autocast(device_type="cuda", dtype=torch.bfloat16)
        return autocast(device_type="cuda", dtype=torch.float16)

    def _apply_projection(self, x: torch.Tensor, projection) -> torch.Tensor:
        """Apply a projection that may be a Linear module or a bare parameter matrix."""
        if projection is None:
            return x
        if isinstance(projection, torch.nn.Module):
            return projection(x)
        if isinstance(projection, torch.Tensor):
            return x @ projection
        if callable(projection):
            return projection(x)
        raise TypeError(f"Unsupported projection type: {type(projection)}")

    def _encode_long_text(self, text_batch: List[str]) -> torch.Tensor:
        all_features = torch.zeros(len(text_batch), self.feature_dim, device=self.device, dtype=torch.float32)
        if len(text_batch) == 0:
            return all_features

        segment_length = self.max_length - 2
        global_segments: List[List[int]] = []
        text_indices: List[int] = []

        # Tokenize without truncation; long texts are split into segments and averaged.
        full_encodings = self.tokenizer(
            text_batch,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        for text_idx, input_ids in enumerate(full_encodings):
            if not input_ids:
                # Empty text still gets BOS/EOS, since an all-pad input breaks CLIP pooling.
                padded = [bos, eos] + [pad] * (self.max_length - 2)
                global_segments.append(padded)
                text_indices.append(text_idx)
                continue

            for offset in range(0, len(input_ids), segment_length):
                seg = input_ids[offset : offset + segment_length]
                padded = [bos] + seg + [eos]
                padded = padded + [pad] * (self.max_length - len(padded))
                global_segments.append(padded[: self.max_length])
                text_indices.append(text_idx)

        segment_counts = torch.zeros(len(text_batch), device=self.device, dtype=torch.float32)

        with torch.no_grad(), self._amp_context():
            for start in range(0, len(global_segments), self.config.TEXT_SEGMENT_BATCH_SIZE):
                batch_segments = global_segments[start : start + self.config.TEXT_SEGMENT_BATCH_SIZE]
                batch_indices = text_indices[start : start + self.config.TEXT_SEGMENT_BATCH_SIZE]

                input_ids = torch.tensor(batch_segments, dtype=torch.long, device=self.device)
                attention_mask = (input_ids != pad).long()

                # get_text_features can return a model output instead of a tensor on some
                # transformers versions, so the pooled output is projected explicitly. No return_dict:
                # recent versions no longer accept it, and older ones return a model output by default.
                text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = text_outputs.pooler_output
                batch_features = self._apply_projection(pooled, self.model.text_projection).float()

                index = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
                all_features.scatter_add_(
                    0,
                    index.unsqueeze(1).expand_as(batch_features),
                    batch_features,
                )
                segment_counts.scatter_add_(
                    0,
                    index,
                    torch.ones(batch_features.shape[0], device=self.device, dtype=torch.float32),
                )

        valid = segment_counts > 0
        all_features[valid] /= segment_counts[valid].unsqueeze(1)
        return all_features

    def _load_image_rgb(self, path: Path) -> Optional[Image.Image]:
        try:
            with Image.open(path) as img:
                return img.convert("RGB").copy()
        except Exception as exc:
            logging.warning(f"Cannot load image {path}: {exc}")
            return None

    def _encode_images(self, img_paths: List[str], base_dir: Path) -> torch.Tensor:
        batch_size = len(img_paths)
        image_features = torch.zeros(batch_size, self.feature_dim, device=self.device, dtype=torch.float32)

        valid_images: List[Image.Image] = []
        original_indices: List[int] = []

        for idx, raw_path in enumerate(img_paths):
            if raw_path is None or pd.isna(raw_path) or str(raw_path).strip() == "":
                continue
            full_path = base_dir / str(raw_path)
            img = self._load_image_rgb(full_path)
            if img is not None:
                valid_images.append(img)
                original_indices.append(idx)

        if not valid_images:
            return image_features

        with torch.no_grad(), self._amp_context():
            for start in range(0, len(valid_images), self.config.IMAGE_BATCH_SIZE):
                sub_images = valid_images[start : start + self.config.IMAGE_BATCH_SIZE]
                sub_indices = original_indices[start : start + self.config.IMAGE_BATCH_SIZE]

                try:
                    inputs = self.processor(images=sub_images, return_tensors="pt").to(self.device)
                    vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                    pooled = vision_outputs.pooler_output
                    features = self._apply_projection(pooled, self.model.visual_projection).float()
                    index = torch.tensor(sub_indices, dtype=torch.long, device=self.device)
                    image_features.index_copy_(0, index, features)
                except Exception as exc:
                    logging.error(f"CLIP image batch failed, keeping zeros for this mini-batch: {exc}")

        return image_features

    def process_batch(self, batch: pd.DataFrame, image_base_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids = batch["i"].to_numpy(dtype=np.int64, copy=True)
        text_batch = batch["text"].fillna("NULL").astype(str).tolist()
        img_batch_paths = batch["img"].fillna("").astype(str).tolist()

        with torch.no_grad():
            text_features = self._encode_long_text(text_batch)
            image_features = self._encode_images(img_batch_paths, image_base_dir)

            missing_img_mask = image_features.norm(dim=1) == 0
            if missing_img_mask.any():
                image_features[missing_img_mask] = text_features[missing_img_mask]

            text_features = F.normalize(text_features, p=2, dim=1)
            image_features = F.normalize(image_features, p=2, dim=1)

            features_cat = torch.cat([text_features, image_features], dim=1).cpu().float().numpy()
            features_rep = torch.cat([text_features, text_features], dim=1).cpu().float().numpy()

        rand_tensor = torch.randn(len(batch), self.feature_dim * 2, generator=self.cpu_generator, dtype=torch.float32)
        features_ran = F.normalize(rand_tensor, p=2, dim=1).numpy()

        return ids, features_cat, features_rep, features_ran


def open_shard_memmaps(tmp_dir: Path, prefix: str, rank: int, rows: int, hidden_size: int):
    ids_mm = np.lib.format.open_memmap(
        tmp_dir / f"{prefix}_rank{rank}_ids.npy",
        mode="w+",
        dtype=np.int64,
        shape=(rows,),
    )
    cat_mm = np.lib.format.open_memmap(
        tmp_dir / f"{prefix}_rank{rank}_cat.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows, hidden_size),
    )
    rep_mm = np.lib.format.open_memmap(
        tmp_dir / f"{prefix}_rank{rank}_rep.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows, hidden_size),
    )
    ran_mm = np.lib.format.open_memmap(
        tmp_dir / f"{prefix}_rank{rank}_ran.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows, hidden_size),
    )
    return ids_mm, cat_mm, rep_mm, ran_mm


def flush_memmaps(*arrays) -> None:
    for arr in arrays:
        arr.flush()


def merge_one_kind(
    tmp_dir: Path,
    output_path: Path,
    prefix: str,
    kind: str,
    total_items: int,
    total_rows: int,
    hidden_size: int,
    world_size: int,
    config: Config,
) -> None:
    final_path = output_path / f"{prefix}_feat_{kind}.npy"
    logging.info(f"Merging {kind} -> {final_path}")

    final_mm = np.lib.format.open_memmap(
        final_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_items + 1, hidden_size),
    )
    final_mm[:] = 0.0

    for r in range(world_size):
        r_start, r_end = shard_bounds(total_rows, r, world_size)
        r_rows = r_end - r_start
        if r_rows <= 0:
            continue

        ids = np.load(tmp_dir / f"{prefix}_rank{r}_ids.npy", mmap_mode="r")
        shard = np.load(tmp_dir / f"{prefix}_rank{r}_{kind}.npy", mmap_mode="r")

        for off in range(0, r_rows, config.MERGE_CHUNK_SIZE):
            end = min(off + config.MERGE_CHUNK_SIZE, r_rows)
            ids_chunk = np.asarray(ids[off:end], dtype=np.int64)
            values = shard[off:end]
            valid = (ids_chunk >= 0) & (ids_chunk <= total_items)
            if not np.all(valid):
                bad = ids_chunk[~valid][:10]
                logging.warning(f"Found invalid ids in rank {r}: {bad}")
            final_mm[ids_chunk[valid]] = values[valid]

    final_mm.flush()
    del final_mm


def generate_features_for_attribute(
    extractor: MultimodalFeatureExtractor,
    attr_csv_path: Path,
    img_dir: Path,
    output_dir: Path,
    total_rows: int,
    total_items: int,
    prefix: str,
    rank: int,
    world_size: int,
) -> None:
    config = extractor.config
    hidden_size = extractor.feature_dim * 2

    tmp_dir = output_dir / "_tmp_clip_ddp_shards" / prefix
    if rank == 0:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    start, end = shard_bounds(total_rows, rank, world_size)
    shard_rows = end - start
    logging.info(
        f"Processing {attr_csv_path.name}, prefix={prefix}, rows=[{start}, {end}), shard_rows={shard_rows}"
    )

    ids_mm, cat_mm, rep_mm, ran_mm = open_shard_memmaps(tmp_dir, prefix, rank, shard_rows, hidden_size)

    filled = 0
    iterator = iter_csv_shard(attr_csv_path, start, shard_rows, config.BATCH_SIZE)
    pbar = tqdm(
        total=shard_rows,
        desc=f"rank{rank}:{prefix}",
        disable=(rank != 0),
    )

    for batch in iterator:
        if batch.empty:
            continue

        ids, cat, rep, ran = extractor.process_batch(batch, img_dir)
        n = len(ids)
        if filled + n > shard_rows:
            raise RuntimeError(f"Shard overflow: filled={filled}, n={n}, shard_rows={shard_rows}")

        ids_mm[filled : filled + n] = ids
        cat_mm[filled : filled + n] = cat
        rep_mm[filled : filled + n] = rep
        ran_mm[filled : filled + n] = ran
        filled += n
        pbar.update(n)

    pbar.close()
    flush_memmaps(ids_mm, cat_mm, rep_mm, ran_mm)
    del ids_mm, cat_mm, rep_mm, ran_mm

    if filled != shard_rows:
        raise RuntimeError(f"Rank {rank} processed rows mismatch: filled={filled}, expected={shard_rows}")

    logging.info(f"Rank {rank} finished shard for prefix={prefix}")
    barrier()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("cat", "rep", "ran"):
            merge_one_kind(
                tmp_dir=tmp_dir,
                output_path=output_dir,
                prefix=prefix,
                kind=kind,
                total_items=total_items,
                total_rows=total_rows,
                hidden_size=hidden_size,
                world_size=world_size,
                config=config,
            )
        if not config.KEEP_SHARDS:
            shutil.rmtree(tmp_dir)
        logging.info(f"Merged prefix={prefix} done")

    barrier()


def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract multimodal CLIP features.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name, e.g. GDELT")
    parser.add_argument("--data-root", type=str, default=".", help="Contains <dataset>/entity_attr.csv and <dataset>/relation_attr.csv")
    parser.add_argument("--image-root", type=str, default=None, help="Contains <dataset>/img_entity and <dataset>/img_relation (default: --data-root)")
    parser.add_argument("--output-root", type=str, default=None, help="Features go to <output-root>/<dataset>/npy (default: --data-root)")
    parser.add_argument("--pretrained-model", type=str, default=config.PRETRAINED_MODEL, help="CLIP model name or local path")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="CSV rows per batch, per rank")
    parser.add_argument("--image-batch-size", type=int, default=config.IMAGE_BATCH_SIZE, help="Image encoding mini-batch size, per rank")
    parser.add_argument("--text-segment-batch-size", type=int, default=config.TEXT_SEGMENT_BATCH_SIZE)
    parser.add_argument("--precision", type=str, default=config.PRECISION, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--keep-shards", action="store_true", help="Keep the per-rank temporary shard files")
    return parser.parse_args()


def main() -> None:
    config = Config()
    args = parse_args(config)

    rank, world_size, local_rank, device = setup_distributed()
    configure_logging(rank)

    config.PRETRAINED_MODEL = args.pretrained_model
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_BATCH_SIZE = args.image_batch_size
    config.TEXT_SEGMENT_BATCH_SIZE = args.text_segment_batch_size
    config.PRECISION = args.precision
    config.KEEP_SHARDS = args.keep_shards

    seed_everything(config.SEED, rank)

    if rank == 0:
        logging.info(f"world_size={world_size}, precision={config.PRECISION}")
    logging.info(f"local_rank={local_rank}, device={device}")

    extractor = MultimodalFeatureExtractor(config=config, device=device, rank=rank)

    dataset_path = Path(args.data_root) / args.dataset
    img_path = Path(args.image_root or args.data_root) / args.dataset
    output_path = Path(args.output_root or args.data_root) / args.dataset / "npy"
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
    barrier()

    entity_csv = dataset_path / "entity_attr.csv"
    relation_csv = dataset_path / "relation_attr.csv"

    e_total_rows, e_max_i = get_csv_stats_distributed(entity_csv, config, rank, device)
    generate_features_for_attribute(
        extractor=extractor,
        attr_csv_path=entity_csv,
        img_dir=img_path / "img_entity",
        output_dir=output_path,
        total_rows=e_total_rows,
        total_items=e_max_i,
        prefix="e",
        rank=rank,
        world_size=world_size,
    )

    r_total_rows, r_max_i = get_csv_stats_distributed(relation_csv, config, rank, device)
    generate_features_for_attribute(
        extractor=extractor,
        attr_csv_path=relation_csv,
        img_dir=img_path / "img_relation",
        output_dir=output_path,
        total_rows=r_total_rows,
        total_items=r_max_i,
        prefix="r",
        rank=rank,
        world_size=world_size,
    )

    barrier()
    if rank == 0:
        logging.info("Done")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
