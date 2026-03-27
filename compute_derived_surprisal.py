"""
Compute surprisal at the juke point for each item in derived.json,
for both the GP sentence and the control.

Outputs to cache/derived_surprisal_{model}_{precision}.json
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from gps import Model
from huggingface_hub import login


MODELS = {
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "70b": "meta-llama/Llama-3.1-70B-Instruct",
}


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        help="Model size: " + ", ".join(f"{k} ({v})" for k, v in MODELS.items()),
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Load in fp16 instead of 4-bit",
    )
    args = parser.parse_args()

    MODEL_ID = MODELS[args.model]
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    base_path = os.path.dirname(os.path.abspath(__file__))
    slug = model_slug(MODEL_ID)
    precision = "fp16" if args.fp16 else "4bit"
    cache_path = os.path.join(
        base_path, "cache", f"derived_surprisal_{slug}_{precision}.json"
    )
    dataset_path = os.path.join(base_path, "data", "derived.json")

    log(f"Model: {MODEL_ID} ({precision})")
    log(f"Cache: {cache_path}")
    model = Model(MODEL_ID, quantize=not args.fp16)

    with open(dataset_path) as f:
        all_data = json.load(f)

    items = [e for e in all_data if e.get("control")]
    log(f"Items with controls: {len(items)}")

    # Load existing cache
    results = {}
    try:
        with open(cache_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    bos_offset = 1 if model.tokenizer.bos_token_id is not None else 0

    for i, item in enumerate(items):
        prefix = item["prefix"]
        prefix_len = len(
            model.tokenizer(prefix, add_special_tokens=False).input_ids
        )
        juke_pos = prefix_len

        cache_key = item["sentence"]
        if cache_key in results:
            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(items)}] cached")
            continue

        # Compute surprisal for GP sentence
        gp_stats = model.compute_sentence_metrics(item["sentence"])
        if juke_pos < len(gp_stats):
            gp_rec = gp_stats[juke_pos]
            gp_surprisal = gp_rec["surprisal"]
            gp_next_token = gp_rec["next_token"]
            gp_rank = gp_rec["rank"]
            gp_entropy = gp_rec["entropy"]
        else:
            gp_surprisal = None
            gp_next_token = None
            gp_rank = None
            gp_entropy = None

        # Compute surprisal for control
        ctrl_stats = model.compute_sentence_metrics(item["control"])
        if juke_pos < len(ctrl_stats):
            ctrl_rec = ctrl_stats[juke_pos]
            ctrl_surprisal = ctrl_rec["surprisal"]
            ctrl_next_token = ctrl_rec["next_token"]
            ctrl_rank = ctrl_rec["rank"]
            ctrl_entropy = ctrl_rec["entropy"]
        else:
            ctrl_surprisal = None
            ctrl_next_token = None
            ctrl_rank = None
            ctrl_entropy = None

        results[cache_key] = {
            "prefix": prefix,
            "type": item["type"],
            "source": item["source"],
            "juke_position": juke_pos,
            "gp": {
                "sentence": item["sentence"],
                "surprisal": gp_surprisal,
                "next_token": gp_next_token,
                "rank": gp_rank,
                "entropy": gp_entropy,
            },
            "control": {
                "sentence": item["control"],
                "surprisal": ctrl_surprisal,
                "next_token": ctrl_next_token,
                "rank": ctrl_rank,
                "entropy": ctrl_entropy,
            },
        }

        if (i + 1) % 10 == 0 or i == 0:
            delta = (gp_surprisal or 0) - (ctrl_surprisal or 0)
            log(
                f"  [{i+1}/{len(items)}] {item['type']:5} "
                f"GP={gp_surprisal:6.2f} ctrl={ctrl_surprisal:6.2f} "
                f"Δ={delta:+.2f}  {prefix[:50]}"
            )

        # Save periodically
        if (i + 1) % 25 == 0:
            with open(cache_path, "w") as f:
                json.dump(results, f)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log(f"\nSaved {len(results)} results to {cache_path}")

    by_type = defaultdict(lambda: {"gp": [], "ctrl": [], "delta": []})
    for r in results.values():
        if r["gp"]["surprisal"] is not None and r["control"]["surprisal"] is not None:
            t = r["type"]
            gs = r["gp"]["surprisal"]
            cs = r["control"]["surprisal"]
            by_type[t]["gp"].append(gs)
            by_type[t]["ctrl"].append(cs)
            by_type[t]["delta"].append(gs - cs)

    log(f"\n{'Type':>10} {'N':>4} {'GP surp':>10} {'Ctrl surp':>10} {'Δ (GP-ctrl)':>12}")
    log("-" * 52)
    for t in sorted(by_type):
        d = by_type[t]
        n = len(d["gp"])
        log(
            f"{t:>10} {n:>4} "
            f"{np.mean(d['gp']):>10.2f} "
            f"{np.mean(d['ctrl']):>10.2f} "
            f"{np.mean(d['delta']):>+12.2f}"
        )

    all_gp = [s for d in by_type.values() for s in d["gp"]]
    all_ctrl = [s for d in by_type.values() for s in d["ctrl"]]
    all_delta = [s for d in by_type.values() for s in d["delta"]]
    log(
        f"{'ALL':>10} {len(all_gp):>4} "
        f"{np.mean(all_gp):>10.2f} "
        f"{np.mean(all_ctrl):>10.2f} "
        f"{np.mean(all_delta):>+12.2f}"
    )
