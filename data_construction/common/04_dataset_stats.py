"""Print a summary of the constructed CSV files of each dataset."""

import argparse
import os

import pandas as pd


def analyze_dataset(dataset_path):
    entity_path = os.path.join(dataset_path, "entity_attr.csv")
    relation_path = os.path.join(dataset_path, "relation_attr.csv")
    edge_list_path = os.path.join(dataset_path, "edge_list.csv")

    df_entity = pd.read_csv(entity_path)
    df_relation = pd.read_csv(relation_path)
    df_edge = pd.read_csv(edge_list_path)

    num_nodes = len(df_entity)
    node_images = df_entity["img"].dropna().loc[df_entity["img"] != ""].count()

    num_edges = len(df_edge)
    edge_images = df_relation["img"].dropna().loc[df_relation["img"] != ""].count()

    if "ts" in df_edge.columns:
        min_ts = df_edge["ts"].min()
        max_ts = df_edge["ts"].max()
        ts_range = max_ts - min_ts
    else:
        min_ts = max_ts = ts_range = "N/A"

    print(f"\nDataset: {os.path.basename(os.path.normpath(dataset_path))}")
    print("=" * 40)
    print(f"Nodes:             {num_nodes}")
    print(f"Edges:             {num_edges}")
    print("-" * 40)
    print(f"Node images:       {node_images}")
    print(f"Edge images:       {edge_images}")
    print(f"Node image ratio:  {node_images / num_nodes}")
    print(f"Edge image ratio:  {edge_images / num_edges}")
    print("-" * 40)
    print(f"Min timestamp:     {min_ts}")
    print(f"Max timestamp:     {max_ts}")
    print(f"Timestamp range:   {ts_range}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Print node, edge, image and timestamp counts.")
    parser.add_argument("--data-root", default=".", help="Directory containing one folder per dataset")
    parser.add_argument("--datasets", nargs="+", default=["GDELT", "GMCT", "Yelp", "amazon_ind", "amazon_off"])
    args = parser.parse_args()

    for dataset_name in args.datasets:
        analyze_dataset(os.path.join(args.data_root, dataset_name))


if __name__ == "__main__":
    main()
