"""
Prepend experiment: does cueing "garden path sentence" modulate juke-point surprisal?

Design: 2 (sentence type: GPS vs alternative) x 2 (prepend vs bare) x 33 unstacked GPS items.
DV: surprisal at the juke point (token position = prefix_length).

No dropout — baseline model only.
"""

import json
import os

from gps import Model
from huggingface_hub import login


def log(msg):
    print(msg, flush=True)


PREPEND = "Consider this garden path sentence: "


def metrics_at_position(model, sentence, position):
    """Return surprisal at a specific token position."""
    stats = model.compute_sentence_metrics(sentence, k=1)
    if position < len(stats):
        return {
            "surprisal": stats[position]["surprisal"],
            "rank": stats[position]["rank"],
        }
    return None


if __name__ == "__main__":
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    base_path = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_path, "cache", "compute_prepend_cache.json")
    dataset_path = os.path.join(base_path, "data/data.json")

    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    model = Model(MODEL_ID)

    gps_data = json.load(open(dataset_path))

    results = {}
    try:
        with open(cache_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    # 33 unstacked GPS items only
    items = [item for item in gps_data if item.get("type") != "CENTER_EMBED"]
    log(f"Unstacked GPS items: {len(items)}")

    meta = []
    for item in items:
        prefix = item["prefix"]
        prefix_length = len(model.tokenizer(prefix, add_special_tokens=False).input_ids)
        prepend_prefix_length = len(
            model.tokenizer(PREPEND + prefix, add_special_tokens=False).input_ids
        )
        meta.append(
            {
                "sentence": item["sentence"],
                "control": item["control"],
                "prefix": prefix,
                "position_bare": prefix_length,
                "position_prepend": prepend_prefix_length,
            }
        )

    # 4 conditions: (gps, alt) x (bare, prepend)
    conditions = [
        ("gps_bare", "sentence", False),
        ("gps_prepend", "sentence", True),
        ("alt_bare", "control", False),
        ("alt_prepend", "control", True),
    ]

    total = len(meta) * len(conditions)
    done = 0

    for cond_key, sent_field, do_prepend in conditions:
        log(f"\nCondition: {cond_key}")
        for m in meta:
            sent = m[sent_field]
            full_sent = PREPEND + sent if do_prepend else sent
            position = m["position_prepend"] if do_prepend else m["position_bare"]

            cache_key = f"{cond_key}:{m['sentence']}"  # always key by GPS sentence

            if cache_key in results:
                done += 1
                log(f"  [{done}/{total}] cached: {sent[:50]}...")
                continue

            r = metrics_at_position(model, full_sent, position)
            done += 1
            if r:
                results[cache_key] = {
                    "surprisal": r["surprisal"],
                    "rank": r["rank"],
                    "position": position,
                    "sentence_used": full_sent,
                }
                log(
                    f"  [{done}/{total}] {cond_key}: surprisal={r['surprisal']:.2f} rank={r['rank']} | {sent[:50]}..."
                )
            else:
                log(
                    f"  [{done}/{total}] {cond_key}: SKIPPED (position out of range) | {sent[:50]}..."
                )

            with open(cache_path, "w") as f:
                json.dump(results, f)

    # ------------------------------------------------------------------ #
    #  Analysis                                                           #
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 60)
    log("ANALYSIS: Prepend x Sentence Type — Juke-Point Surprisal")
    log("=" * 60)

    summary = {}
    for cond_key, _, _ in conditions:
        vals = []
        for m in meta:
            cache_key = f"{cond_key}:{m['sentence']}"
            if cache_key in results:
                vals.append(results[cache_key]["surprisal"])
        import numpy as np

        arr = np.array(vals)
        summary[cond_key] = {
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }
        log(
            f"  {cond_key:>15}: n={len(vals):>3}  mean={arr.mean():.2f}  std={arr.std():.2f}  median={np.median(arr):.2f}"
        )

    # Paired differences
    log("\nPaired differences (prepend - bare):")
    for sent_type in ["gps", "alt"]:
        bare_key = f"{sent_type}_bare"
        prep_key = f"{sent_type}_prepend"
        diffs = []
        for m in meta:
            bk = f"{bare_key}:{m['sentence']}"
            pk = f"{prep_key}:{m['sentence']}"
            if bk in results and pk in results:
                diffs.append(results[pk]["surprisal"] - results[bk]["surprisal"])
        diffs = np.array(diffs)
        from scipy import stats as sp_stats

        t, p = sp_stats.ttest_rel(
            [
                results[f"{prep_key}:{m['sentence']}"]["surprisal"]
                for m in meta
                if f"{prep_key}:{m['sentence']}" in results
                and f"{bare_key}:{m['sentence']}" in results
            ],
            [
                results[f"{bare_key}:{m['sentence']}"]["surprisal"]
                for m in meta
                if f"{prep_key}:{m['sentence']}" in results
                and f"{bare_key}:{m['sentence']}" in results
            ],
        )
        log(f"  {sent_type}: mean_diff={diffs.mean():+.2f}  t={t:.3f}  p={p:.4f}")

    log("\nPaired differences (gps - alt):")
    for cond in ["bare", "prepend"]:
        gps_key = f"gps_{cond}"
        alt_key = f"alt_{cond}"
        gps_vals, alt_vals = [], []
        for m in meta:
            gk = f"{gps_key}:{m['sentence']}"
            ak = f"{alt_key}:{m['sentence']}"
            if gk in results and ak in results:
                gps_vals.append(results[gk]["surprisal"])
                alt_vals.append(results[ak]["surprisal"])
        diffs = np.array(gps_vals) - np.array(alt_vals)
        t, p = sp_stats.ttest_rel(gps_vals, alt_vals)
        log(f"  {cond}: mean_diff={diffs.mean():+.2f}  t={t:.3f}  p={p:.4f}")

    summary_path = os.path.join(base_path, "cache", "prepend_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {summary_path}")
