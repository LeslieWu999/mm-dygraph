"""Command line arguments for all tasks."""
from __future__ import annotations

import argparse

import torch

from utils.modeling import BACKBONE_REGISTRY, WRAPPER_NAMES

# Per-model hyperparameters that differ from the parser defaults. Explicit command line values win.
_BEST_CONFIGS = {
    "JODIE": {"num_neighbors": 10, "num_layers": 1},
    "DyRep": {"num_neighbors": 10, "num_layers": 1},
    "TGN": {"num_neighbors": 10, "num_layers": 1},
    "CAWN": {"num_neighbors": 32, "sample_neighbor_strategy": "time_interval_aware", "position_feat_dim": 768},
    "FreeDyG": {"max_input_sequence_length": 32},
    "TPNet": {"use_random_projection": True},
    "Liquid": {"num_neighbors": 10, "num_layers": 1},
}
_TASK_BEST_CONFIGS = {
    "link_prediction": {"TGN": {"num_neighbors": 5, "max_input_sequence_length": 12}},
    "edge_classification": {"GraphMixer": {"dropout": 0.5}},
}


def get_link_prediction_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments for link prediction training and node retrieval evaluation."""
    parser = _build_common_parser("Link prediction and node retrieval")
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive"],
        help="negative edge sampling strategy of node retrieval",
    )
    parser.add_argument(
        "--retrieval_evaluation_protocol",
        type=str,
        default="sampled",
        choices=["sampled", "recurrence_aware"],
        help="sampled metrics, or additionally stratify queries into repeated and novel directed edges",
    )
    parser.add_argument(
        "--num_negative_candidates",
        type=int,
        default=100,
        help="number of sampled destination candidates per retrieval query",
    )
    return _parse_args(parser, "link_prediction", argv)


def get_edge_classification_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_args(_build_common_parser("Edge classification"), "edge_classification", argv)


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    group = parser.add_argument_group("experiment")
    group.add_argument("--dataset_name", type=str, required=True, help="dataset directory name under the data root, or a path to it")
    group.add_argument("--model_name", type=str, required=True, choices=sorted(BACKBONE_REGISTRY), help="temporal backbone (SALoM is 'Liquid')")
    group.add_argument("--wrapper_name", type=str, default="concat", choices=WRAPPER_NAMES, help="multimodal fusion wrapper")
    group.add_argument(
        "--use_feature",
        type=str,
        default="text+image",
        help=(
            "'+'-joined modalities out of rand, text, image, image_zero, image_mean, image_learned"
        ),
    )
    group.add_argument("--data_root", type=str, default=None, help="dataset root directory, defaults to $MMDG_DATA_ROOT")
    group.add_argument("--output_dir", type=str, default=".", help="directory for saved_models/, saved_results/ and logs/")
    group.add_argument("--seeds", type=str, default=None, help="comma-separated random seeds, overrides --num_runs")
    group.add_argument("--num_runs", type=int, default=3, help="number of runs, using seeds 0..num_runs-1")
    group.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    group.add_argument("--offload_features", action="store_true", help="keep feature tables in host memory (same results, less GPU memory)")
    group.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    group.add_argument(
        "--no-load_best_configs",
        dest="load_best_configs",
        action="store_false",
        help="do not apply the per-model default hyperparameters",
    )

    group = parser.add_argument_group("training")
    group.add_argument("--batch_size", type=int, default=256, help="batch size")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    group.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    group.add_argument("--num_epochs", type=int, default=200, help="maximum number of epochs")
    group.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "RMSprop"], help="optimizer")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument("--patience", type=int, default=5, help="early stopping patience, in evaluations")
    group.add_argument("--val_ratio", type=float, default=0.15, help="validation ratio")
    group.add_argument("--test_ratio", type=float, default=0.15, help="test ratio")
    group.add_argument("--test_interval_epochs", type=int, default=1, help="evaluate every this many epochs")

    group = parser.add_argument_group("backbone")
    group.add_argument("--num_neighbors", type=int, default=20, help="neighbors sampled for each node")
    group.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="historical neighbor sampling strategy",
    )
    group.add_argument("--time_scaling_factor", type=float, default=1e-6, help="sampling bias of time_interval_aware sampling")
    group.add_argument("--num_walk_heads", type=int, default=8, help="CAWN: walk encoder attention heads")
    group.add_argument("--num_heads", type=int, default=2, help="attention heads")
    group.add_argument("--num_layers", type=int, default=2, help="number of temporal layers")
    group.add_argument("--walk_length", type=int, default=1, help="CAWN: random walk length")
    group.add_argument("--time_gap", type=int, default=2000, help="GraphMixer: neighbor time gap")
    group.add_argument("--time_feat_dim", type=int, default=100, help="time encoding dimension")
    group.add_argument("--position_feat_dim", type=int, default=172, help="CAWN: position encoding dimension")
    group.add_argument("--patch_size", type=int, default=1, help="DyGFormer: patch size")
    group.add_argument("--channel_embedding_dim", type=int, default=50, help="DyGFormer: channel embedding dimension")
    group.add_argument("--max_input_sequence_length", type=int, default=48, help="maximum length of a node's input sequence")

    group = parser.add_argument_group("TPNet")
    group.add_argument("--use_random_projection", action="store_true", help="enable the random projection module (on by default for TPNet)")
    group.add_argument("--rp_num_layer", type=int, default=3, help="maximum hop of the temporal walk matrices")
    group.add_argument("--rp_time_decay_weight", type=float, default=1e-6, help="time decay weight (lambda)")
    group.add_argument("--rp_dim_factor", type=int, default=10, help="controls the random projection dimension")
    group.add_argument("--rp_use_matrix", action="store_true", help="explicitly maintain the walk matrices")
    group.add_argument("--rp_not_scale", action="store_true", help="do not scale the projection inner product")
    group.add_argument("--enforce_dim", type=int, default=-1, help="force the random projection dimension")
    group.add_argument("--encode_not_rp", action="store_true", help="drop pairwise features in the encoder")

    group = parser.add_argument_group("fusion baselines")
    group.add_argument("--gate_per_dimension", type=int, default=1, choices=[0, 1], help="gated_fusion: 1 = per-dimension gate, 0 = scalar gate")
    group.add_argument("--mattn_num_queries", type=int, default=1, help="modality_attention: learnable queries, output dim = num_queries * feat_dim")
    group.add_argument("--mattn_num_heads", type=int, default=4, help="modality_attention: attention heads")
    group.add_argument("--unigraph2_num_experts", type=int, default=8, help="unigraph2: number of experts")
    group.add_argument("--unigraph2_mask_ratio", type=float, default=0.75, help="unigraph2: mask ratio of the reconstruction loss, 0 disables it")
    group.add_argument("--unigraph2_recon_weight", type=float, default=0.1, help="unigraph2: weight of the reconstruction loss")
    return parser


def _parse_args(parser: argparse.ArgumentParser, task_name: str, argv: list[str] | None) -> argparse.Namespace:
    known_args, _ = parser.parse_known_args(argv)
    if known_args.load_best_configs:
        parser.set_defaults(**_BEST_CONFIGS.get(known_args.model_name, {}))
        parser.set_defaults(**_TASK_BEST_CONFIGS[task_name].get(known_args.model_name, {}))
    args = parser.parse_args(argv)

    if args.val_ratio <= 0 or args.test_ratio <= 0 or (args.val_ratio + args.test_ratio) >= 1:
        parser.error(f"expected val_ratio and test_ratio in (0, 1) with a sum below 1, got {args.val_ratio} and {args.test_ratio}")
    if args.wrapper_name.startswith("tst_mf") and args.model_name != "DyGFormer":
        parser.error(f"--wrapper_name {args.wrapper_name} requires --model_name DyGFormer")
    if hasattr(args, "num_negative_candidates") and args.num_negative_candidates <= 0:
        parser.error(f"expected num_negative_candidates > 0, got {args.num_negative_candidates}")
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    args.use_feature = args.use_feature.strip()
    args.seeds = _resolve_seeds(parser, args.seeds, args.num_runs)
    return args


def _resolve_seeds(parser: argparse.ArgumentParser, raw_seeds: str | None, num_runs: int) -> list[int]:
    if raw_seeds is None:
        if num_runs <= 0:
            parser.error(f"expected num_runs > 0, got {num_runs}")
        return list(range(num_runs))
    try:
        seeds = [int(item) for item in raw_seeds.replace(",", " ").split()]
    except ValueError:
        parser.error(f"invalid --seeds '{raw_seeds}'")
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error(f"--seeds must contain distinct integers, got '{raw_seeds}'")
    return seeds
