"""
Selectivity experiment: early-layer ablation on derived.json.

For each model, ablate attn or mlp at layers 0, 1, 2 individually.
Measure surprisal at the juke point for GP sentence and control.

Hypothesis: if early layers matter for garden-path processing,
ablating them should selectively affect GP surprisal more than control.

Models:
  qwen7b   — Qwen/Qwen2.5-7B-Instruct (bf16)
  mistral  — mistralai/Mistral-7B-Instruct-v0.3 (bf16)
  qwen14b  — Qwen/Qwen2.5-14B-Instruct (bf16)
  llama8b  — meta-llama/Llama-3.1-8B-Instruct (bf16)
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from gps import Model
from huggingface_hub import login


MODELS = {
    "qwen7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B",
    "mistral7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral7b": "mistralai/Mistral-7B-v0.3",
    "gemma9b": "google/gemma-2-9b",
    "qwen14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "llama8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B",
}

ABLATION_LAYERS = [0, 1, 2]
COMPONENTS = ["attn", "mlp"]
TOP_K = 5


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ #
#  Tokenizer verification                                             #
# ------------------------------------------------------------------ #
def verify_tokenization(model, items):
    """Verify that prefix tokenization aligns correctly for every item.

    Checks:
    1. tokenize(prefix) is a true prefix of tokenize(sentence)
    2. tokenize(prefix) is a true prefix of tokenize(control)
    3. The juke-point token (first token after prefix) differs between
       sentence and control
    """
    tokenizer = model.tokenizer
    bos_offset = 1 if tokenizer.bos_token_id is not None else 0
    failures = []

    for i, item in enumerate(items):
        prefix = item["prefix"]
        sentence = item["sentence"]
        control = item["control"]

        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        sent_ids = tokenizer(sentence, add_special_tokens=False).input_ids
        ctrl_ids = tokenizer(control, add_special_tokens=False).input_ids

        p_len = len(prefix_ids)

        # Check prefix alignment
        if sent_ids[:p_len] != prefix_ids:
            failures.append(
                (i, "sentence prefix mismatch",
                 tokenizer.convert_ids_to_tokens(prefix_ids),
                 tokenizer.convert_ids_to_tokens(sent_ids[:p_len + 1]))
            )
        if ctrl_ids[:p_len] != prefix_ids:
            failures.append(
                (i, "control prefix mismatch",
                 tokenizer.convert_ids_to_tokens(prefix_ids),
                 tokenizer.convert_ids_to_tokens(ctrl_ids[:p_len + 1]))
            )

        # Check divergence at juke point
        if (sent_ids[:p_len] == prefix_ids and ctrl_ids[:p_len] == prefix_ids
                and len(sent_ids) > p_len and len(ctrl_ids) > p_len):
            if sent_ids[p_len] == ctrl_ids[p_len]:
                sent_tok = tokenizer.convert_ids_to_tokens([sent_ids[p_len]])[0]
                ctrl_tok = tokenizer.convert_ids_to_tokens([ctrl_ids[p_len]])[0]
                failures.append(
                    (i, f"same juke token: {sent_tok}",
                     item["prefix"], item["sentence"][:60])
                )

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        help="Model: " + ", ".join(f"{k} ({v})" for k, v in MODELS.items()),
    )
    args = parser.parse_args()

    MODEL_ID = MODELS[args.model]
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    base_path = os.path.dirname(os.path.abspath(__file__))
    slug = model_slug(MODEL_ID)
    cache_path = os.path.join(
        base_path, "cache", f"selectivity_{slug}_bf16.json"
    )
    dataset_path = os.path.join(base_path, "data", "derived.json")

    log(f"Model: {MODEL_ID} (bf16)")
    log(f"Cache: {cache_path}")
    model = Model(MODEL_ID, quantize=False, dtype="bf16")

    with open(dataset_path) as f:
        all_data = json.load(f)

    # Filter to items with controls
    items = [e for e in all_data if e.get("control")]
    log(f"Items: {len(items)}")

    # ---- Tokenizer verification ----
    log("\nTokenizer verification...")
    failures = verify_tokenization(model, items)
    if failures:
        log(f"  FAILED: {len(failures)} issues")
        for idx, msg, a, b in failures[:10]:
            log(f"    [{idx}] {msg}")
            log(f"      {a}")
            log(f"      {b}")
        log("  Aborting. Fix tokenization issues before running.")
        sys.exit(1)
    else:
        log(f"  PASSED: all {len(items)} items verified")

    # ---- Build sentence metadata ----
    sent_meta = []
    for item in items:
        prefix = item["prefix"]
        prefix_length = len(
            model.tokenizer(prefix, add_special_tokens=False).input_ids
        )
        for sent_type, sent_field in [("gps", "sentence"), ("ctrl", "control")]:
            sent_meta.append({
                "sentence": item[sent_field],
                "gps_sentence": item["sentence"],
                "sent_type": sent_type,
                "prefix": prefix,
                "prefix_length": prefix_length,
                "position": prefix_length,
                "type": item["type"],
            })

    # ---- Conditions ----
    conditions = ["none"] + [
        f"{comp}_L{layer}"
        for layer in ABLATION_LAYERS
        for comp in COMPONENTS
    ]

    log(f"\nSentences (GPS + ctrl): {len(sent_meta)}")
    log(f"Ablation layers: {ABLATION_LAYERS}")
    log(f"Conditions: {len(conditions)} (1 baseline + {len(ABLATION_LAYERS)}×2 ablations)")

    # ---- Load cache ----
    results = {}
    try:
        with open(cache_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    # ---- Run ----
    for cond_idx, condition in enumerate(conditions):
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
            cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
            results.setdefault(cache_key, {})
            if condition not in results[cache_key]:
                stats = model.compute_sentence_metrics(m["sentence"], k=TOP_K)
                pos = m["position"]
                if pos < len(stats):
                    rec = stats[pos]
                    results[cache_key][condition] = {
                        "rank": rec["rank"],
                        "surprisal": rec["surprisal"],
                        "entropy": rec["entropy"],
                        "position": pos,
                        "topk": [[tok, prob] for tok, prob in rec["topk"]],
                    }

        with open(cache_path, "w") as f:
            json.dump(results, f)

    model.set_ablation(None)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Analysis                                                           #
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log(f"SELECTIVITY: Early-Layer Ablation — {MODEL_ID}")
    log("=" * 70)

    # Baseline
    baseline = {"gps": [], "ctrl": []}
    for m in sent_meta:
        cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
        if cache_key in results and "none" in results[cache_key]:
            baseline[m["sent_type"]].append(results[cache_key]["none"]["surprisal"])

    log(f"\nBaseline  GPS: {np.mean(baseline['gps']):>8.2f} ± {np.std(baseline['gps'])/np.sqrt(len(baseline['gps'])):.2f}")
    log(f"Baseline ctrl: {np.mean(baseline['ctrl']):>8.2f} ± {np.std(baseline['ctrl'])/np.sqrt(len(baseline['ctrl'])):.2f}")
    log(f"Baseline   Δ : {np.mean(baseline['gps']) - np.mean(baseline['ctrl']):>+8.2f}")

    # Per-condition
    log(f"\n{'condition':>10} | {'GPS':>8} {'Δ GPS':>8} | {'ctrl':>8} {'Δ ctrl':>8} | {'GP-ctrl':>8} {'selectivity':>12}")
    log("-" * 82)

    baseline_gps = np.mean(baseline["gps"])
    baseline_ctrl = np.mean(baseline["ctrl"])
    baseline_gap = baseline_gps - baseline_ctrl

    for layer in ABLATION_LAYERS:
        for comp in COMPONENTS:
            condition = f"{comp}_L{layer}"
            gps_surps, ctrl_surps = [], []
            for m in sent_meta:
                cache_key = f"{m['sent_type']}:{m['gps_sentence']}"
                if cache_key in results and condition in results[cache_key]:
                    val = results[cache_key][condition]["surprisal"]
                    if m["sent_type"] == "gps":
                        gps_surps.append(val)
                    else:
                        ctrl_surps.append(val)

            if gps_surps and ctrl_surps:
                gps_m = np.mean(gps_surps)
                ctrl_m = np.mean(ctrl_surps)
                d_gps = gps_m - baseline_gps
                d_ctrl = ctrl_m - baseline_ctrl
                gap = gps_m - ctrl_m
                selectivity = gap - baseline_gap  # positive = ablation widens the GP effect
                log(
                    f"{condition:>10} | {gps_m:>8.2f} {d_gps:>+8.2f} | "
                    f"{ctrl_m:>8.2f} {d_ctrl:>+8.2f} | "
                    f"{gap:>+8.2f} {selectivity:>+12.2f}"
                )

    # By type breakdown
    for gp_type in ["NPS", "MVRR", "NPVP"]:
        type_meta = [m for m in sent_meta if m["type"] == gp_type]
        if not type_meta:
            continue

        log(f"\n--- {gp_type} ({len(type_meta) // 2} items) ---")

        bl_gps = []
        bl_ctrl = []
        for m in type_meta:
            ck = f"{m['sent_type']}:{m['gps_sentence']}"
            if ck in results and "none" in results[ck]:
                if m["sent_type"] == "gps":
                    bl_gps.append(results[ck]["none"]["surprisal"])
                else:
                    bl_ctrl.append(results[ck]["none"]["surprisal"])

        if not bl_gps or not bl_ctrl:
            continue

        bl_gps_m = np.mean(bl_gps)
        bl_ctrl_m = np.mean(bl_ctrl)
        bl_gap = bl_gps_m - bl_ctrl_m
        log(f"  Baseline: GPS={bl_gps_m:.2f}  ctrl={bl_ctrl_m:.2f}  Δ={bl_gap:+.2f}")

        for layer in ABLATION_LAYERS:
            for comp in COMPONENTS:
                condition = f"{comp}_L{layer}"
                gs, cs = [], []
                for m in type_meta:
                    ck = f"{m['sent_type']}:{m['gps_sentence']}"
                    if ck in results and condition in results[ck]:
                        if m["sent_type"] == "gps":
                            gs.append(results[ck][condition]["surprisal"])
                        else:
                            cs.append(results[ck][condition]["surprisal"])
                if gs and cs:
                    gap = np.mean(gs) - np.mean(cs)
                    sel = gap - bl_gap
                    log(f"  {condition:>10}: GPS={np.mean(gs):.2f}  ctrl={np.mean(cs):.2f}  Δ={gap:+.2f}  sel={sel:+.2f}")

    log(f"\nCache saved to {cache_path}")
