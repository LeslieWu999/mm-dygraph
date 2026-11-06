"""Reduce the paths in the `image` field of a JSONL file to bare file names."""

import argparse

import orjson


def _strip_entry(entry):
    """An entry is a path, or a [path, description] pair."""
    if isinstance(entry, str):
        return entry.rsplit("/", 1)[-1]
    return [entry[0].rsplit("/", 1)[-1], *entry[1:]]


def process_jsonl_fast(input_file, output_file):
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            data = orjson.loads(line)

            if "image" in data and isinstance(data["image"], list):
                data["image"] = [_strip_entry(entry) for entry in data["image"]]

            f_out.write(orjson.dumps(data) + b"\n")


def main():
    parser = argparse.ArgumentParser(description="Strip directories from the image field of a JSONL file.")
    parser.add_argument("--input", default="edges.jsonl")
    parser.add_argument("--output", default="edges_new.jsonl")
    args = parser.parse_args()
    process_jsonl_fast(args.input, args.output)


if __name__ == "__main__":
    main()
