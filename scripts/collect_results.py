"""Collects all results under saved_results/ into per-run and per-group CSV files and one JSON file.

Metric names are not hardcoded: every "<split> metrics" dict and "<split> loss" scalar becomes a column.

    python -m scripts.collect_results
    python -m scripts.collect_results --task link_prediction --dataset amazon_ind --out results_amazon_ind

Output, with the default prefix "results":
    results_runs.csv     one row per run (seed)
    results_grouped.csv  one row per dataset x model x wrapper x feature, with mean, std and n
    results.json         both of the above, including the missing seeds of each group
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# saved_results/<model_dir>/<dataset>/<task>__<wrapper>__<model>__feature-xxx[__negative-xxx][__seed-N].json
FILENAME_RE = re.compile(r"^(?P<task>[a-z_]+?)(?P<summary>_summary)?__(?P<rest>.+)\.json$")


def parse_filename(path: Path):
    """Parses a result file name into metadata, or returns None if it does not match."""
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    tokens = match.group("rest").split("__")
    if len(tokens) < 2:
        return None

    info = {
        "task": match.group("task"),
        "is_summary": bool(match.group("summary")),
        "wrapper": tokens[0],
        "model": tokens[1],
        "feature": None,
        "negative_sample_strategy": None,
        "seed": None,
    }
    for token in tokens[2:]:
        if token.startswith("feature-"):
            info["feature"] = token[len("feature-"):]
        elif token.startswith("negative-"):
            info["negative_sample_strategy"] = token[len("negative-"):]
        elif token.startswith("seed-"):
            info["seed"] = int(token[len("seed-"):])

    # The dataset is everything between the model directory and the file name.
    parts = path.parts
    try:
        idx = parts.index("saved_results")
        info["dataset"] = "/".join(parts[idx + 2: -1]) or None
    except ValueError:
        info["dataset"] = None
    return info


def flatten_metrics(payload: dict) -> dict:
    """Flattens {"test loss": x, "test metrics": {...}} to {"test.loss": x, "test.average_precision": ...}."""
    flat = {}

    def leaf_value(value):
        if isinstance(value, dict) and "mean" in value:
            return value["mean"]
        return value

    for key, value in payload.items():
        if key.endswith(" metrics") and isinstance(value, dict):
            split = key[: -len(" metrics")]
            for metric_name, metric_value in value.items():
                flat[f"{split}.{metric_name}"] = leaf_value(metric_value)
        elif key.endswith(" loss"):
            split = key[: -len(" loss")]
            flat[f"{split}.loss"] = leaf_value(value)
        elif isinstance(value, (int, float)):
            flat[key] = value
    return flat


def mean_std(values):
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return None, None, 0
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0, len(clean)
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)   # sample standard deviation
    return mean, math.sqrt(var), len(clean)


def collect(root: Path, args):
    runs = []
    for path in sorted(root.rglob("*.json")):
        info = parse_filename(path)
        if info is None:
            continue
        if info["is_summary"]:
            continue                      # statistics are recomputed here
        if info["seed"] is None:
            continue                      # aggregate file
        if args.task and info["task"] != args.task:
            continue
        if args.dataset and info["dataset"] != args.dataset:
            continue
        if args.model and info["model"] != args.model:
            continue
        if args.wrapper and info["wrapper"] != args.wrapper:
            continue

        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
        if args.since and mtime < args.since:
            continue

        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
            continue

        row = dict(info)
        row.pop("is_summary")
        row["finished_at"] = mtime.strftime("%Y-%m-%d %H:%M")
        row["path"] = str(path)
        row.update(flatten_metrics(payload))
        runs.append(row)
    return runs


GROUP_KEYS = ["task", "dataset", "model", "wrapper", "feature", "negative_sample_strategy"]
META_KEYS = GROUP_KEYS + ["seed", "finished_at", "path"]


def group_runs(runs, expected_seeds):
    groups = defaultdict(list)
    for row in runs:
        groups[tuple(row.get(k) for k in GROUP_KEYS)].append(row)

    metric_keys = sorted({k for row in runs for k in row if k not in META_KEYS})
    grouped = []
    for key, rows in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        seeds = sorted(r["seed"] for r in rows)
        missing = sorted(set(expected_seeds) - set(seeds)) if expected_seeds else []
        entry = dict(zip(GROUP_KEYS, key))
        entry["n_seeds"] = len(rows)
        entry["seeds"] = ",".join(str(s) for s in seeds)
        entry["missing_seeds"] = ",".join(str(s) for s in missing)
        entry["complete"] = not missing
        for metric in metric_keys:
            mean, std, n = mean_std([r.get(metric) for r in rows])
            entry[f"{metric}.mean"] = mean
            entry[f"{metric}.std"] = std
            entry[f"{metric}.n"] = n
        grouped.append(entry)
    return grouped, metric_keys


def write_csv(path: Path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(grouped, metric_keys, highlight):
    """Prints the main metrics; the CSV files have all of them."""
    preferred = [m for m in ("test.average_precision", "test.roc_auc", "test.H1",
                             "test.F_weight", "test.F_macro") if m in metric_keys]
    if highlight:
        preferred = [m for m in highlight if m in metric_keys] or preferred
    if not preferred:
        preferred = metric_keys[:2]

    header = f"{'dataset':<12}{'model':<11}{'wrapper':<22}{'n':>2}  "
    header += "  ".join(f"{m.replace('test.', ''):>22}" for m in preferred)
    print(header)
    print("-" * len(header))
    for entry in grouped:
        line = f"{str(entry['dataset']):<12}{str(entry['model']):<11}{str(entry['wrapper']):<22}{entry['n_seeds']:>2}  "
        cells = []
        for metric in preferred:
            mean, std = entry.get(f"{metric}.mean"), entry.get(f"{metric}.std")
            cells.append("                     -" if mean is None
                         else f"{mean:>10.4f} ± {std:<9.4f}")
        line += "  ".join(cells)
        if not entry["complete"]:
            line += f"   [missing seeds {entry['missing_seeds']}]"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Collect the experiment results under saved_results/")
    parser.add_argument("--root", default="saved_results", help="results root directory")
    parser.add_argument("--out", default="results", help="output file prefix")
    parser.add_argument("--task", help="only this task: link_prediction, edge_classification or node_retrieval")
    parser.add_argument("--dataset", help="only this dataset")
    parser.add_argument("--model", help="only this backbone")
    parser.add_argument("--wrapper", help="only this fusion wrapper")
    parser.add_argument("--since", help="only runs finished after this time: YYYY-MM-DD or 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--expected-seeds", default="0,1,2", help="expected seeds, used to flag missing runs; empty disables the check")
    parser.add_argument("--metrics", help="comma-separated metrics to print, chosen automatically by default")
    args = parser.parse_args()

    if args.since:
        fmt = "%Y-%m-%d %H:%M" if " " in args.since else "%Y-%m-%d"
        args.since = dt.datetime.strptime(args.since, fmt)

    expected_seeds = [int(s) for s in args.expected_seeds.split(",") if s.strip()] if args.expected_seeds else []
    highlight = [m.strip() for m in args.metrics.split(",")] if args.metrics else None

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Directory {root} not found; use --root to set the results directory.")

    runs = collect(root, args)
    if not runs:
        sys.exit("No matching results. Results are only written when a run finishes.")

    grouped, metric_keys = group_runs(runs, expected_seeds)

    run_columns = META_KEYS + metric_keys
    group_columns = (GROUP_KEYS + ["n_seeds", "seeds", "missing_seeds", "complete"]
                     + [f"{m}.{stat}" for m in metric_keys for stat in ("mean", "std", "n")])

    runs_path = Path(f"{args.out}_runs.csv")
    grouped_path = Path(f"{args.out}_grouped.csv")
    json_path = Path(f"{args.out}.json")

    write_csv(runs_path, sorted(runs, key=lambda r: [str(r.get(k)) for k in GROUP_KEYS + ["seed"]]), run_columns)
    write_csv(grouped_path, grouped, group_columns)
    json_path.write_text(json.dumps(
        {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "n_runs": len(runs), "n_groups": len(grouped),
         "runs": runs, "groups": grouped},
        indent=2, ensure_ascii=False))

    print_table(grouped, metric_keys, highlight)
    incomplete = [g for g in grouped if not g["complete"]]
    print(f"\n{len(runs)} runs in {len(grouped)} groups" + (f", {len(incomplete)} groups have missing seeds" if incomplete else ""))
    print(f"runs -> {runs_path}\ngroups -> {grouped_path}\nfull -> {json_path}")


if __name__ == "__main__":
    main()
