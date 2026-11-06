"""Score image-text pairs with a vision-language model behind a vLLM server.

For every record with non-empty `text` and `image`, each downloaded image is sent to an
OpenAI-compatible chat endpoint, which returns relevance and consistency scores. Result ids
are `line<L>_img<I>`, where L and I are 1-based positions in the input file; 02_construct_graph.py
must therefore read the same file line by line. Pairs that keep failing get scores of -1.
"""

import argparse
import asyncio
import base64
import datetime
import json
import os
import re
from urllib.parse import urlparse

import aiohttp

PROMPT_TEMPLATE = """### Task
Evaluate the relationship between the provided image and the following text.

### Text
{text}{extra_description}

### Evaluation Criteria
We want to measure two different aspects:
1. Relevance: How related is the image to the text? (e.g., they depict the same item or topic. The image might provide extra information not mentioned in the text, which is completely fine and should not lower the relevance score).
2. Consistency: How strictly does the image match the exact details explicitly mentioned in the text? (It's okay if this is lower than relevance when the image has extra unmentioned features or slight deviations).

### Output Instructions
Please explain your reasoning in a few sentences first.
After your reasoning, provide four scores, each ranging from 0 to 10 (you can use decimals like 7.3):
- relevance_score
- relevance_confidence (Your confidence in the Relevance Score)
- consistency_score
- consistency_confidence (Your confidence in the Consistency Score)

### Output Format
Output your final response STRICTLY in the following format. Do NOT copy the placeholder numbers, make your own judgments. Do not add any text after the JSON block.

Reasoning:
[Your reasoning here]

```json
{{"relevance_score": [0-10 score], "relevance_confidence": [0-10 score], "consistency_score": [0-10 score], "consistency_confidence": [0-10 score]}}
```
"""  # noqa: E501

TEMPERATURE = 0.2
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 60


def get_filename(path_or_url):
    if re.match(r"^[a-zA-Z]+://", path_or_url):
        parsed = urlparse(path_or_url)
        return os.path.basename(parsed.path)
    else:
        return os.path.basename(path_or_url)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def evaluate_pair(session, args, pair_id, filename, text, local_img_path, semaphore, image_description=""):
    async with semaphore:
        extra_desc_str = ""
        if image_description:
            extra_desc_str = f"\n\n### Extra Image Description\n{image_description}"

        prompt = PROMPT_TEMPLATE.format(text=text, extra_description=extra_desc_str)

        try:
            base64_image = encode_image(local_img_path)
            ext = os.path.splitext(local_img_path)[1].lower()
            mime_type = "image/jpeg"
            if ext == ".png":
                mime_type = "image/png"
            elif ext == ".webp":
                mime_type = "image/webp"

            img_data_uri = f"data:{mime_type};base64,{base64_image}"
        except Exception as e:
            print(f"Failed to read image {local_img_path} for {pair_id} ({filename}): {e}")
            return None

        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_uri}},
                    ],
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        required_keys = {
            "relevance_score",
            "relevance_confidence",
            "consistency_score",
            "consistency_confidence",
        }

        attempts = 0
        last_error = ""
        while attempts < args.max_attempts:
            attempts += 1
            try:
                async with session.post(args.api_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    if response.status != 200:
                        last_error = f"HTTP {response.status}: {await response.text()}"
                        await asyncio.sleep(2)
                        continue

                    resp_json = await response.json()
                    content = resp_json["choices"][0]["message"]["content"]

                    # Prefer the fenced JSON block, fall back to the outermost braces.
                    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        match = re.search(r"(\{[\s\S]*\})", content)
                        if match:
                            json_str = match.group(1)
                        else:
                            last_error = "no JSON object in model output"
                            continue

                    scores = json.loads(json_str)

                    if not required_keys.issubset(scores.keys()):
                        last_error = f"missing keys, found {list(scores.keys())}"
                        continue

                    return {
                        "id": pair_id,
                        "filename": filename,
                        "text": text,
                        "scores": scores,
                        "raw_output": content,
                    }

            except asyncio.TimeoutError:
                last_error = "timeout"
            except json.JSONDecodeError:
                last_error = "invalid JSON in model output"
            except Exception as e:
                last_error = str(e)

        print(f"Failed to evaluate {pair_id} ({filename}) after {args.max_attempts} attempts: {last_error}")
        return {
            "id": pair_id,
            "filename": filename,
            "text": text,
            "scores": {
                "relevance_score": -1,
                "relevance_confidence": -1,
                "consistency_score": -1,
                "consistency_confidence": -1,
            },
            "raw_output": f"FAILED_AFTER_{args.max_attempts}_ATTEMPTS",
        }


async def run(args):
    print(f"Loading data from {args.input_file}...")
    tasks_data = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                text = data.get("text", "")
                images = data.get("image", [])

                if not text or not images:
                    continue

                for img_idx, img in enumerate(images, 1):
                    img_url = ""
                    image_description = ""

                    # An entry is either a path or a [path, description] pair.
                    if isinstance(img, list):
                        img_url = img[0]
                        if len(img) > 1:
                            image_description = img[1]
                    else:
                        img_url = img

                    filename = get_filename(img_url)
                    local_img_path = os.path.join(args.image_dir, filename)

                    if not os.path.exists(local_img_path):
                        continue

                    pair_id = f"line{line_idx}_img{img_idx}"
                    tasks_data.append((pair_id, filename, text, local_img_path, image_description))

            except json.JSONDecodeError:
                continue

    print(f"Prepared {len(tasks_data)} image-text pairs for evaluation (concurrency {args.concurrency}).")

    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(evaluate_pair(session, args, pair_id, filename, text, local_img_path, semaphore, image_description))
            for pair_id, filename, text, local_img_path, image_description in tasks_data
        ]

        completed = 0
        total = len(tasks)
        buffer = []

        with open(args.output_file, "w", encoding="utf-8") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed += 1

                if result:
                    buffer.append(result)

                if len(buffer) >= args.concurrency or (completed == total and buffer):
                    for r in buffer:
                        f.write(json.dumps(r) + "\n")
                    buffer = []

                if completed % args.concurrency == 0 or completed == total:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{current_time} Progress: {completed}/{total}")

            f.flush()

    print(f"Evaluation complete. Results saved to {args.output_file}.")


def main():
    parser = argparse.ArgumentParser(description="Score image-text pairs through a vLLM server.")
    parser.add_argument("--input-file", required=True, help="Input JSONL file (nodes or edges)")
    parser.add_argument("--image-dir", required=True, help="Flat directory containing the downloaded images")
    parser.add_argument("--output-file", required=True, help="Output JSONL file with scores")
    parser.add_argument("--concurrency", type=int, default=128, help="Number of concurrent requests to vLLM")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions", help="Chat completions endpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name as served by vLLM")
    parser.add_argument("--max-attempts", type=int, default=2, help="Attempts per image-text pair before scoring it -1")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
