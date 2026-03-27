"""
Compile derived.json from compiled.json + data.json.

Inclusion criteria:
  1. From compiled.json: all items that have a control sentence, EXCEPT
     SAP benchmark NPS items whose verb has a cloze sentence-complement
     bias (cloze_SZMbias) >= 0.5 or missing.  High-bias verbs already
     favour the sentential-complement parse, so the sentence is not
     genuinely ambiguous at the prefix.
  2. From data.json: all items that have a control sentence, EXCEPT
     CENTER_EMBED constructions (not a garden-path type).

Verb bias source: SAP Benchmark norming data (verbbias.csv),
  column cloze_SZMbias, condition GPE_NPS.
"""

import json
import math
from pathlib import Path

import pandas as pd

DATA = Path("data")
VERB_BIAS_THRESHOLD = 0.5


def load_sap_verb_bias():
    """Return {item_number: cloze_SZMbias} for NPS items."""
    df = pd.read_csv(DATA / "sap_benchmark" / "verbbias.csv")
    nps = df[df["coef"] == "GPE_NPS"]
    bias = {}
    for _, row in nps.iterrows():
        val = row["cloze_SZMbias"]
        bias[int(row["item"])] = val if not (isinstance(val, float) and math.isnan(val)) else None
    return bias


def load_sap_item_numbers():
    """Return {ambiguous_sentence: item_number} for NPS items."""
    df = pd.read_csv(DATA / "sap_benchmark" / "items_ClassicGP.csv")
    nps = df[df["condition"] == "NPS_UAMB"]
    return {row["ambiguous"]: int(row["item"]) for _, row in nps.iterrows()}


def main():
    with open(DATA / "compiled.json") as f:
        compiled = json.load(f)
    with open(DATA / "data.json") as f:
        data = json.load(f)

    sap_bias = load_sap_verb_bias()
    sap_items = load_sap_item_numbers()

    entries = []
    seen = set()

    # --- compiled.json ---
    excluded_sap = 0
    for item in compiled:
        if not item.get("control"):
            continue

        # Filter SAP NPS by verb bias
        if item["source"] == "sap_benchmark" and item["type"] == "NPS":
            item_num = sap_items.get(item["sentence"])
            if item_num is None:
                excluded_sap += 1
                continue
            bias = sap_bias.get(item_num)
            if bias is None or bias >= VERB_BIAS_THRESHOLD:
                excluded_sap += 1
                continue

        entry = {
            "sentence": item["sentence"],
            "prefix": item["prefix"],
            "type": item["type"],
            "source": item["source"],
            "control": item["control"],
            "unambiguous": item.get("unambiguous", []),
        }
        entries.append(entry)
        seen.add(item["sentence"])

    print(f"From compiled.json: {len(entries)} items ({excluded_sap} SAP NPS excluded)")

    # --- data.json ---
    added = 0
    for item in data:
        if not item.get("control"):
            continue
        if item["type"] == "CENTER_EMBED":
            continue
        if item["sentence"] in seen:
            continue

        entry = {
            "sentence": item["sentence"],
            "prefix": item["prefix"],
            "type": item["type"],
            "source": item["source"],
            "control": item["control"],
            "unambiguous": item.get("unambiguous", []),
        }
        entries.append(entry)
        seen.add(item["sentence"])
        added += 1

    print(f"From data.json: {added} items (excluding CENTER_EMBED and duplicates)")
    print(f"Total: {len(entries)} items")

    out_path = DATA / "derived.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
