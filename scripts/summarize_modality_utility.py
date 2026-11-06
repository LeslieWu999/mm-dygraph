"""Summarises the output of analyze_modality_utility.py.

For each dataset, task and modality: the mean utility, its 95% confidence interval, the share of
instances with positive utility, and the per-seed means. The interval comes from a two-level
bootstrap that resamples the seeds and then the instances within each seed, so it reflects the
variation across seeds and not only the sample size.

    python -m scripts.summarize_modality_utility --utility_dir saved_results/_utility --datasets GDELT Yelp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SPLITS = [("lp_tr", "LP tr."), ("lp_in", "LP in."), ("ec", "EC")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utility_dir", type=Path, default=Path("saved_results/_utility"))
    parser.add_argument("--datasets", nargs="+", default=["GDELT", "GMCT", "amazon_ind", "amazon_off", "Yelp"])
    parser.add_argument("--wrapper_name", default="tst_mf")
    parser.add_argument("--model_name", default="DyGFormer")
    parser.add_argument("--modalities", nargs="+", default=["text", "image"], help="modality names, in the order of --use_feature")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print(f"{'dataset':<11}{'task':<7}{'modality':<9}{'mean':>9}{'95% CI':>22}{'positive':>10}{'n':>8}  per-seed means")
    for dataset in args.datasets:
        for split, label in SPLITS:
            files = sorted(args.utility_dir.glob(f"utility__{split}__{dataset}__{args.wrapper_name}__{args.model_name}__seed-*.npz"))
            if not files:
                print(f"{dataset:<11}{label:<7}  (no results)")
                continue
            per_seed = [np.load(file)["delta"] for file in files]
            for m, modality in enumerate(args.modalities):
                deltas = [seed_delta[:, m] for seed_delta in per_seed]
                pooled = np.concatenate(deltas)
                boots = []
                for _ in range(args.num_bootstrap):
                    picked = rng.integers(0, len(deltas), len(deltas))
                    boots.append(np.mean(np.concatenate([deltas[i][rng.integers(0, len(deltas[i]), len(deltas[i]))] for i in picked])))
                low, high = np.percentile(boots, [2.5, 97.5])
                print(
                    f"{dataset:<11}{label:<7}{modality:<9}{pooled.mean():>+9.4f}   [{low:+.4f}, {high:+.4f}]"
                    f"{np.mean(pooled > 0):>10.3f}{len(pooled):>8}  {[round(float(x.mean()), 4) for x in deltas]}"
                )


if __name__ == "__main__":
    main()
