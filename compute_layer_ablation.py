"""
Per-layer, per-component ablation experiment.

For each layer L (0-27) and each component C \in {attn, mlp}:
  - Skip that one component at that one layer (zero ablation)
  - Measure surprisal, rank, and top-5 continuations at the juke point
  - Compare to the clean baseline

"""

import argparse
import json
import os
import re

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
        help="Model size to run: " + ", ".join(f"{k} ({v})" for k, v in MODELS.items()),
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Load model in fp16 instead of 4-bit (requires more VRAM)",
    )
    args = parser.parse_args()

    MODEL_ID = MODELS[args.model]
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    base_path = os.path.dirname(os.path.abspath(__file__))
    slug = model_slug(MODEL_ID)
    precision = "fp16" if args.fp16 else "4bit"
    cache_path = os.path.join(base_path, "cache", f"layer_ablation_{slug}_{precision}.json")
    dataset_path = os.path.join(base_path, "data/data.json")

    log(f"Model: {MODEL_ID} ({precision})")
    log(f"Cache: {cache_path}")
    model = Model(MODEL_ID, quantize=not args.fp16)

    gps_data = json.load(open(dataset_path))

    results = {}
    try:
        with open(cache_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    # 33 unstacked GPS items + their alternatives
    items = [item for item in gps_data if item.get("type") != "CENTER_EMBED"]
    sent_meta = []
    for item in items:
        prefix = item["prefix"]
        prefix_length = len(model.tokenizer(prefix, add_special_tokens=False).input_ids)
        for sent_type, sent_field in [("gps", "sentence"), ("alt", "control")]:
            sent_meta.append(
                {
                    "sentence": item[sent_field],
                    "gps_sentence": item["sentence"],
                    "sent_type": sent_type,
                    "prefix": prefix,
                    "prefix_length": prefix_length,
                    "position": prefix_length,
                }
            )

    n_layers = len(model.model.model.layers)
    components = ["attn", "mlp"]
    # Include "none" as the clean baseline
    conditions = ["none"] + [
        f"{comp}_L{layer}" for layer in range(n_layers) for comp in components
    ]

    log(f"Sentences (GPS + alt): {len(sent_meta)}")
    log(f"Layers: {n_layers}")
    log(f"Conditions: {len(conditions)} (1 baseline + {n_layers}×2 ablations)")

    TOP_K = 5

    for cond_idx, condition in enumerate(conditions):
        # Check if all entries cached
        all_cached = all(
            f"{m['sent_type']}:{m['gps_sentence']}" in results
            and condition in results.get(f"{m['sent_type']}:{m['gps_sentence']}", {})
            for m in sent_meta
        )

        if all_cached:
            log(f"  [{cond_idx + 1}/{len(conditions)}] {condition}: cached")
            continue

        # Set up ablation
        if condition == "none":
            model.set_ablation(None)
        else:
            comp, layer_str = condition.split("_L")
            layer_idx = int(layer_str)
            model.set_ablation(
                layer_idx,
                skip_attn=(comp == "attn"),
                skip_mlp=(comp == "mlp"),
            )

        log(f"  [{cond_idx + 1}/{len(conditions)}] {condition}: running...")

        for m in sent_meta:
            sentence = m["sentence"]
            position = m["position"]

            cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
            results.setdefault(cache_key, {})
            if condition not in results[cache_key]:
                stats = model.compute_sentence_metrics(sentence, k=TOP_K)
                if position < len(stats):
                    rec = stats[position]
                    results[cache_key][condition] = {
                        "rank": rec["rank"],
                        "surprisal": rec["surprisal"],
                        "entropy": rec["entropy"],
                        "position": position,
                        "topk": [[tok, prob] for tok, prob in rec["topk"]],
                    }

        with open(cache_path, "w") as f:
            json.dump(results, f)

    # Clean up
    model.set_ablation(None)

    # ------------------------------------------------------------------ #
    #  Analysis                                                           #
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("ANALYSIS: Per-Layer Ablation - Juke-Point Metrics")
    log("=" * 70)

    # Get baseline values
    baseline_gps_surp = []
    baseline_alt_surp = []
    for m in sent_meta:
        cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
        if cache_key in results and "none" in results[cache_key]:
            val = results[cache_key]["none"]["surprisal"]
            if m["sent_type"] == "gps":
                baseline_gps_surp.append(val)
            else:
                baseline_alt_surp.append(val)

    log(
        f"\nBaseline - GPS surprisal: {np.mean(baseline_gps_surp):.2f} ± {np.std(baseline_gps_surp) / np.sqrt(len(baseline_gps_surp)):.2f}"
    )
    log(
        f"Baseline - ALT surprisal: {np.mean(baseline_alt_surp):.2f} ± {np.std(baseline_alt_surp) / np.sqrt(len(baseline_alt_surp)):.2f}"
    )

    log(
        f"\n{'condition':>12} | {'GPS surp':>10} {'Δ GPS':>8} | {'ALT surp':>10} {'Δ ALT':>8} | {'GPS rank':>10} {'ALT rank':>10}"
    )
    log("-" * 90)

    for layer in range(n_layers):
        for comp in components:
            condition = f"{comp}_L{layer}"
            gps_surps, alt_surps = [], []
            gps_ranks, alt_ranks = [], []
            for m in sent_meta:
                cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
                if cache_key in results and condition in results[cache_key]:
                    rec = results[cache_key][condition]
                    if m["sent_type"] == "gps":
                        gps_surps.append(rec["surprisal"])
                        gps_ranks.append(rec["rank"])
                    else:
                        alt_surps.append(rec["surprisal"])
                        alt_ranks.append(rec["rank"])

            if gps_surps and alt_surps:
                gps_s = np.mean(gps_surps)
                alt_s = np.mean(alt_surps)
                d_gps = gps_s - np.mean(baseline_gps_surp)
                d_alt = alt_s - np.mean(baseline_alt_surp)
                log(
                    f"{condition:>12} | {gps_s:>10.2f} {d_gps:>+8.2f} | {alt_s:>10.2f} {d_alt:>+8.2f} | {np.mean(gps_ranks):>10.1f} {np.mean(alt_ranks):>10.1f}"
                )

    # Top-5 shift analysis: how many baseline top-5 tokens survive each ablation?
    log(f"\n{'condition':>12} | {'GPS top5 overlap':>16} {'ALT top5 overlap':>16}")
    log("-" * 55)

    for layer in range(n_layers):
        for comp in components:
            condition = f"{comp}_L{layer}"
            gps_overlaps, alt_overlaps = [], []
            for m in sent_meta:
                cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
                if cache_key not in results:
                    continue
                base_rec = results[cache_key].get("none")
                abl_rec = results[cache_key].get(condition)
                if base_rec and abl_rec:
                    base_toks = set(t for t, _ in base_rec["topk"])
                    abl_toks = set(t for t, _ in abl_rec["topk"])
                    overlap = len(base_toks & abl_toks) / len(base_toks)
                    if m["sent_type"] == "gps":
                        gps_overlaps.append(overlap)
                    else:
                        alt_overlaps.append(overlap)

            if gps_overlaps and alt_overlaps:
                log(
                    f"{condition:>12} | {np.mean(gps_overlaps):>14.1%}   {np.mean(alt_overlaps):>14.1%}"
                )

    log(f"\nCache saved to {cache_path}")
