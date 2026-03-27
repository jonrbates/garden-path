"""
Self-priming experiment: does repeating the sentence reduce juke-point surprisal?

Design: 2 (sentence type: GPS vs alternative) x 2 (first vs second occurrence) x 33 unstacked GPS items.
Input: sentence + '\\n' + sentence
DV: surprisal at the juke point in each occurrence.
"""

import json
import os

import numpy as np
from scipy import stats as sp_stats

from gps import Model
from huggingface_hub import login


def log(msg):
    print(msg, flush=True)


def surprisal_at(model, sentence, position):
    """Return surprisal and rank at a specific token position."""
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
    cache_path = os.path.join(base_path, "cache", "compute_selfprime_cache.json")
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

        for sent_field in ["sentence", "control"]:
            sent = item[sent_field]
            concat = sent + "\n" + sent
            # Tokenize the lead-up to the second juke point to get exact position
            lead = sent + "\n" + prefix
            pos_second = len(model.tokenizer(lead, add_special_tokens=False).input_ids)

            meta.append(
                {
                    "gps_sentence": item["sentence"],
                    "sent_field": sent_field,
                    "sent": sent,
                    "prefix": prefix,
                    "pos_first": prefix_length,
                    "pos_second": pos_second,
                    "concat": concat,
                }
            )

    total = len(meta)
    done = 0

    for m in meta:
        sent_type = "gps" if m["sent_field"] == "sentence" else "alt"
        cache_key = f"{sent_type}:{m['gps_sentence']}"

        if cache_key in results:
            done += 1
            log(f"  [{done}/{total}] cached: {m['sent'][:50]}...")
            continue

        r_first = surprisal_at(model, m["concat"], m["pos_first"])
        r_second = surprisal_at(model, m["concat"], m["pos_second"])

        if r_first and r_second:
            results[cache_key] = {
                "first_surprisal": r_first["surprisal"],
                "first_rank": r_first["rank"],
                "second_surprisal": r_second["surprisal"],
                "second_rank": r_second["rank"],
                "pos_first": m["pos_first"],
                "pos_second": m["pos_second"],
            }
            done += 1
            log(
                f"  [{done}/{total}] {sent_type}: 1st={r_first['surprisal']:.2f} 2nd={r_second['surprisal']:.2f} | {m['sent'][:50]}..."
            )
        else:
            done += 1
            log(
                f"  [{done}/{total}] {sent_type}: SKIPPED (position out of range) | {m['sent'][:50]}..."
            )

        with open(cache_path, "w") as f:
            json.dump(results, f)

    # ------------------------------------------------------------------ #
    #  Analysis                                                           #
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 60)
    log("ANALYSIS: Self-Priming — Juke-Point Surprisal (1st vs 2nd)")
    log("=" * 60)

    summary = {}
    for sent_type in ["gps", "alt"]:
        first_vals, second_vals = [], []
        for item in items:
            ck = f"{sent_type}:{item['sentence']}"
            if ck in results:
                first_vals.append(results[ck]["first_surprisal"])
                second_vals.append(results[ck]["second_surprisal"])

        first_arr = np.array(first_vals)
        second_arr = np.array(second_vals)
        diff_arr = second_arr - first_arr

        t, p = sp_stats.ttest_rel(second_vals, first_vals)

        summary[sent_type] = {
            "n": len(first_vals),
            "first_mean": float(first_arr.mean()),
            "second_mean": float(second_arr.mean()),
            "diff_mean": float(diff_arr.mean()),
            "diff_std": float(diff_arr.std()),
            "t": float(t),
            "p": float(p),
        }

        log(f"\n  {sent_type.upper()} (n={len(first_vals)}):")
        log(
            f"    1st occurrence: mean={first_arr.mean():.2f}  std={first_arr.std():.2f}"
        )
        log(
            f"    2nd occurrence: mean={second_arr.mean():.2f}  std={second_arr.std():.2f}"
        )
        log(
            f"    diff (2nd-1st): mean={diff_arr.mean():+.2f}  std={diff_arr.std():.2f}"
        )
        log(f"    paired t-test:  t={t:.3f}  p={p:.4f}")

    # Interaction: is the priming effect larger for GPS than alt?
    log("\nInteraction (GPS priming - Alt priming):")
    gps_diffs, alt_diffs = [], []
    for item in items:
        gk = f"gps:{item['sentence']}"
        ak = f"alt:{item['sentence']}"
        if gk in results and ak in results:
            gps_diffs.append(
                results[gk]["second_surprisal"] - results[gk]["first_surprisal"]
            )
            alt_diffs.append(
                results[ak]["second_surprisal"] - results[ak]["first_surprisal"]
            )
    gps_diffs = np.array(gps_diffs)
    alt_diffs = np.array(alt_diffs)
    interaction = gps_diffs - alt_diffs
    t, p = sp_stats.ttest_rel(gps_diffs, alt_diffs)
    log(
        f"  GPS diff mean: {gps_diffs.mean():+.2f}  Alt diff mean: {alt_diffs.mean():+.2f}"
    )
    log(f"  Interaction:    {interaction.mean():+.2f}  t={t:.3f}  p={p:.4f}")

    summary_path = os.path.join(base_path, "cache", "selfprime_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {summary_path}")
