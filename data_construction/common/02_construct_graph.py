"""Convert scored JSONL into edge_list.csv, relation_attr.csv and entity_attr.csv.

For each node and edge the best image is chosen from the scores of 01_score_images.py. Edges
are sorted by timestamp, timestamps are bucketed and renumbered, and node and edge ids are
assigned from 1 in order of appearance (id 0 is left as padding).
"""

import argparse
import csv
import json
import os
import random

RELEVANCE_THRESHOLD = 6.0


def load_validation_scores(filepath):
    """Load scores as {line index: {image index: result}}; a missing file gives an empty map."""
    scores = {}
    if not filepath or not os.path.exists(filepath):
        return scores

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            parts = data["id"].split("_")
            line_idx = int(parts[0].replace("line", ""))
            img_idx = int(parts[1].replace("img", ""))

            if line_idx not in scores:
                scores[line_idx] = {}
            scores[line_idx][img_idx] = data
    return scores


def select_best_image(image_list, line_idx, scores_map):
    """Return the image with the highest relevance at or above the threshold; ties break randomly."""
    if not image_list or line_idx not in scores_map:
        return ""

    line_scores = scores_map[line_idx]
    candidates = []

    for img_idx, _ in enumerate(image_list, 1):
        if img_idx in line_scores:
            res = line_scores[img_idx]
            rel = res["scores"].get("relevance_score", 0)

            if rel >= RELEVANCE_THRESHOLD:
                candidates.append({
                    "filename": res.get("filename", ""),
                    "rel": rel,
                })

    if not candidates:
        return ""

    random.shuffle(candidates)
    candidates.sort(key=lambda x: x["rel"], reverse=True)

    return candidates[0]["filename"]


def process_to_final_format(args):
    node_scores = load_validation_scores(args.node_validation)
    edge_scores = load_validation_scores(args.edge_validation)

    node_attr_map = {}
    print(f"Reading nodes: {args.input_nodes}")
    with open(args.input_nodes, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            data = json.loads(line)
            best_img = select_best_image(data.get("image", []), idx, node_scores)
            node_attr_map[data["id"]] = {
                "text": data.get("text", ""),
                "img": best_img,
            }

    edges_with_best_img = []
    print(f"Reading edges: {args.input_edges}")
    with open(args.input_edges, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            edge = json.loads(line)
            edge["best_img"] = select_best_image(edge.get("image", []), idx, edge_scores)
            edges_with_best_img.append(edge)

    edges_with_best_img.sort(key=lambda x: x["timestamp"])

    interval = args.bucket_interval
    bucketed_times = sorted(list(set([(e["timestamp"] // interval) * interval for e in edges_with_best_img])))
    ts_map = {t: i for i, t in enumerate(bucketed_times)}

    node_id_map = {}
    next_node_idx = 1
    ordered_nodes = []
    os.makedirs(args.output_dir, exist_ok=True)

    edge_list_path = os.path.join(args.output_dir, "edge_list.csv")
    relation_attr_path = os.path.join(args.output_dir, "relation_attr.csv")
    entity_attr_path = os.path.join(args.output_dir, "entity_attr.csv")

    with open(edge_list_path, "w", encoding="utf-8", newline="") as f_el, \
         open(relation_attr_path, "w", encoding="utf-8", newline="") as f_ra:

        el_writer = csv.writer(f_el, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        ra_writer = csv.writer(f_ra, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

        el_writer.writerow(["u", "r", "i", "ts", "label"])
        ra_writer.writerow(["i", "img", "text"])

        for idx, edge in enumerate(edges_with_best_img, 1):
            u_orig, v_orig = edge["u_id"], edge["v_id"]
            for node_orig in [u_orig, v_orig]:
                if node_orig not in node_id_map:
                    node_id_map[node_orig] = next_node_idx
                    ordered_nodes.append(node_orig)
                    next_node_idx += 1

            u_new, i_new = node_id_map[u_orig], node_id_map[v_orig]
            ts_new = ts_map[(edge["timestamp"] // interval) * interval]

            el_writer.writerow([u_new, idx, i_new, ts_new, edge.get("label", 0)])
            ra_writer.writerow([idx, edge["best_img"], edge.get("text", "")])

    with open(entity_attr_path, "w", encoding="utf-8", newline="") as f_ea:
        ea_writer = csv.writer(f_ea, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        ea_writer.writerow(["i", "img", "text"])
        for node_orig in ordered_nodes:
            attr = node_attr_map.get(node_orig, {"img": "", "text": ""})
            ea_writer.writerow([node_id_map[node_orig], attr["img"], attr["text"]])

    print(f"Done, results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert scored JSONL to the MM-DyGraph CSV format.")
    parser.add_argument("--input-edges", required=True)
    parser.add_argument("--input-nodes", required=True)
    parser.add_argument("--node-validation", required=True, help="Node scores (.jsonl) from 01_score_images.py")
    parser.add_argument("--edge-validation", required=True, help="Edge scores (.jsonl) from 01_score_images.py")
    parser.add_argument("--output-dir", default="./output_csv")
    parser.add_argument("--bucket-interval", type=int, default=900000, help="Time bucket width in milliseconds (15 minutes for GDELT)")

    args = parser.parse_args()
    random.seed(42)
    process_to_final_format(args)


if __name__ == "__main__":
    main()
