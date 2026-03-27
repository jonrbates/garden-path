"""
Compile 5 external garden-path datasets into data.json format.

Output schema:
  sentence:     the ambiguous (garden-path) reading
  prefix:       the sentence up to and including the ambiguous region
  control:      non-GP continuation of the same prefix (string or null)
  unambiguous:  list of structurally disambiguated variants (different prefix)
  type:         construction type (NPS, NPZ, MVRR, NPVP, CENTER_EMBED)
  source:       dataset provenance
"""

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

DATA = Path("data")
OUT = DATA / "compiled.json"

entries = []


def add(sentence, prefix, unambiguous, source, gp_type):
    s = sentence.strip()
    p = prefix.strip()
    unambs = [a.strip() for a in unambiguous if a and a.strip()]
    if not s or not p:
        return
    entries.append({
        "sentence": s,
        "prefix": p,
        "control": None,
        "unambiguous": unambs,
        "type": gp_type,
        "source": source,
    })


# ---------------------------------------------------------------------------
# 1. SAP Benchmark — items_ClassicGP.csv
# ---------------------------------------------------------------------------
def process_sap():
    path = DATA / "sap_benchmark" / "items_ClassicGP.csv"
    if not path.exists():
        print(f"  SKIP {path}")
        return
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            gp_type = cond.split("_")[0]  # NPS, NPZ, MVRR

            amb = row["ambiguous"].strip()
            unamb = row["unambiguous"].strip()
            disamb_pos = int(row["disambPositionAmb"])

            words = amb.split()
            prefix = " ".join(words[:disamb_pos - 1])

            add(amb, prefix, [unamb], "sap_benchmark", gp_type)

    print(f"  SAP Benchmark: done")


# ---------------------------------------------------------------------------
# 2. SyntaxGym — parquet files
# ---------------------------------------------------------------------------
def process_syntaxgym():
    try:
        import pandas as pd
    except ImportError:
        print("  SKIP syntaxgym (no pandas)")
        return

    sg_dir = DATA / "syntaxgym"
    if not sg_dir.exists():
        print(f"  SKIP {sg_dir}")
        return

    suite_map = {
        "npz_ambig": "NPZ",
        "npz_ambig_mod": "NPZ",
        "npz_obj": "NPZ",
        "npz_obj_mod": "NPZ",
        "mvrr": "MVRR",
        "mvrr_mod": "MVRR",
        "center_embed": "CENTER_EMBED",
        "center_embed_mod": "CENTER_EMBED",
    }

    for pfile in sorted(sg_dir.glob("*.parquet")):
        suite_name = pfile.stem
        if suite_name not in suite_map:
            continue

        gp_type = suite_map[suite_name]
        df = pd.read_parquet(pfile)

        for _, row in df.iterrows():
            conds = row["conditions"]
            cond_names = list(conds["condition_name"])
            contents = list(conds["content"])
            regions_list = list(conds["regions"])

            # Identify the most-ambiguous condition
            amb_idx = None
            for i, cn in enumerate(cond_names):
                cl = cn.lower()
                if "ambig" in cl and "unambig" not in cl and "nocomma" in cl:
                    amb_idx = i
                    break
                if "reduced" in cl and "unreduced" not in cl and "ambig" in cl:
                    amb_idx = i
                    break

            if amb_idx is None:
                for i, cn in enumerate(cond_names):
                    cl = cn.lower()
                    if ("ambig" in cl and "unambig" not in cl) or \
                       ("reduced" in cl and "unreduced" not in cl):
                        amb_idx = i
                        break

            if amb_idx is None:
                continue

            amb_sent = contents[amb_idx].strip()

            # Build prefix from regions
            regs = regions_list[amb_idx]
            reg_contents = list(regs["content"])
            non_empty = [c.strip() for c in reg_contents if c.strip()]
            if len(non_empty) >= 2:
                prefix = " ".join(non_empty[:-2])
            else:
                prefix = " ".join(non_empty)

            # All other conditions are unambiguous structural variants
            unambs = []
            for i, c in enumerate(contents):
                if i != amb_idx and c.strip() != amb_sent:
                    unambs.append(c.strip())

            add(amb_sent, prefix, unambs,
                f"syntaxgym/{suite_name}", gp_type)

    print(f"  SyntaxGym: done")


# ---------------------------------------------------------------------------
# 3. Wang SF — stimuli.csv
# ---------------------------------------------------------------------------
def process_wang():
    path = DATA / "wang_sf" / "stimuli.csv"
    if not path.exists():
        print(f"  SKIP {path}")
        return

    type_map = {"S": "NPS", "Z": "NPZ"}

    groups = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["Dataset"], row["Index"], row["Type"])
            groups[key].append(row)

    for (dataset, idx, stype), rows in groups.items():
        gp_type = type_map.get(stype, stype)

        amb_rows = [r for r in rows if r["Ambiguity"] == "A"]
        unamb_rows = [r for r in rows if r["Ambiguity"] == "U"]

        for amb_row in amb_rows:
            amb_raw = amb_row["Stimulus"]
            regions = [r.strip() for r in amb_raw.split("/")]
            amb_sent = " ".join(regions)

            if len(regions) >= 3:
                prefix = " ".join(regions[:-2])
            else:
                prefix = regions[0]

            unambs = []
            for ur in unamb_rows:
                unamb_sent = " ".join(r.strip() for r in ur["Stimulus"].split("/"))
                if unamb_sent != amb_sent:
                    unambs.append(unamb_sent)

            source = f"wang_sf/{dataset.lower()}"
            add(amb_sent, prefix, unambs, source, gp_type)

    print(f"  Wang SF: done")


# ---------------------------------------------------------------------------
# 4. Jurayj GPT2 — nps.tsv, npz.tsv, vawip.tsv
# ---------------------------------------------------------------------------
def read_tsv(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def join_parts(*parts):
    return " ".join(p for p in parts if p)


def process_jurayj():
    base = DATA / "jurayj_gpt2"
    if not base.exists():
        print(f"  SKIP {base}")
        return

    # --- NPS ---
    for row in read_tsv(base / "nps.tsv"):
        subj = row["Subject"].strip()
        ctx = row.get("Context", "").strip()
        amb_verb = row["Ambiguous Verb"].strip()
        unamb_verb = row["Unambiguous Verb"].strip()
        that = row.get("That", "that").strip()
        nps = row["NP/S"].strip()
        ext = row.get("Extension", "").strip()
        disamb = row["Disambiguator"].strip()
        rest = row.get("Rest", "").strip()

        amb_sent = join_parts(subj, ctx, amb_verb, nps, ext, disamb, rest)
        prefix = join_parts(subj, ctx, amb_verb, nps)

        # Unambiguous variants (all alter the prefix)
        alt1 = join_parts(subj, ctx, unamb_verb, that, nps, ext, disamb, rest)
        alt2 = join_parts(subj, ctx, amb_verb, that, nps, ext, disamb, rest)

        add(amb_sent, prefix, [alt1, alt2], "jurayj_gpt2", "NPS")

    # --- NPZ ---
    for row in read_tsv(base / "npz.tsv"):
        start = row["Start"].strip()
        ctx = row.get("Context", "").strip()
        tv = row["Transitive Verb"].strip()
        iv = row["Intransitive Verb"].strip()
        blocker = row.get("Blocker", "").strip()
        comma = row.get("Comma", ",").strip()
        npz = row["NP/Z"].strip()
        ext = row.get("Extension", "").strip()
        verb = row["Verb"].strip()
        rest = row.get("Rest", "").strip()

        amb_sent = join_parts(start, ctx, tv, npz, ext, verb, rest)
        prefix = join_parts(start, ctx, tv, npz)

        alt1 = join_parts(start, ctx, iv + comma, npz, ext, verb, rest)
        alt2 = join_parts(start, ctx, tv, blocker + comma, npz, ext, verb, rest)
        alt3 = join_parts(start, ctx, tv + comma, npz, ext, verb, rest)

        add(amb_sent, prefix, [alt1, alt2, alt3], "jurayj_gpt2", "NPZ")

    # --- VAWIP (MV/RR) ---
    for row in read_tsv(base / "vawip.tsv"):
        start = row["Start"].strip()
        noun = row["Noun"].strip()
        amb_v = row["Ambiguous verb"].strip()
        unamb_v = row["Unambiguous verb"].strip()
        unreduced = row["Unreduced content"].strip()
        rc = row["RC contents"].strip()
        interv = row.get("Intervener", "").strip()
        disamb = row["Disambiguator"].strip()
        end = row.get("End", "").strip()

        amb_sent = join_parts(start, noun, amb_v, rc, interv, disamb, end)
        prefix = join_parts(start, noun, amb_v, rc, interv)

        alt1 = join_parts(start, noun, unamb_v, rc, interv, disamb, end)
        alt2 = join_parts(start, noun, unreduced, amb_v, rc, interv, disamb, end)
        alt3 = join_parts(start, noun, unreduced, unamb_v, rc, interv, disamb, end)

        add(amb_sent, prefix, [alt1, alt2, alt3], "jurayj_gpt2", "MVRR")

    print(f"  Jurayj GPT2: done")


# ---------------------------------------------------------------------------
# 5. Amouyal — NPS, NPVP, reduced relative CSVs
# ---------------------------------------------------------------------------
def process_amouyal():
    base = DATA / "amouyal"
    if not base.exists():
        print(f"  SKIP {base}")
        return

    file_map = {
        "nps_human_base_data.csv": ("NPS", "nps_gp", "nps_nongp"),
        "npvp_human_base_data.csv": ("NPVP", "npvp_gp", "npvp_nongp"),
        "reduced_relative_human_base_data.csv": ("MVRR", "reduced_relative_gp", "reduced_relative_nongp"),
    }

    for fname, (gp_type, gp_label, nongp_label) in file_map.items():
        path = base / fname
        if not path.exists():
            continue

        groups = defaultdict(dict)
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                base_id = row["set_id"].split("_")[0]
                st = row["sent_type"]
                if st not in groups[base_id]:
                    groups[base_id][st] = row

        for base_id, variants in groups.items():
            gp_row = variants.get(gp_label)
            nongp_row = variants.get(nongp_label)
            if not gp_row:
                continue

            sentence = gp_row["sentence"].strip()

            gp_verb = gp_row["gp_verb"].strip()
            reduced_verb = gp_row.get("reduced_verb", "").strip()
            words = sentence.split()

            if reduced_verb:
                # reduced_verb marks the juke point; prefix is everything before it
                rv_idx = None
                for i, w in enumerate(words):
                    if w.strip(".,!?;:") == reduced_verb:
                        rv_idx = i
                        break
                if rv_idx is not None:
                    prefix = " ".join(words[:rv_idx])
                else:
                    prefix = " ".join(words[:len(words) // 2])
            else:
                # NPVP: no reduced_verb; prefix ends at the ambiguous noun/verb
                verb_idx = None
                for i, w in enumerate(words):
                    if w.strip(".,!?;:") == gp_verb:
                        verb_idx = i
                        break
                if verb_idx is not None:
                    prefix = " ".join(words[:verb_idx + 1])
                else:
                    prefix = " ".join(words[:len(words) // 2])

            unambs = []
            if nongp_row:
                unambs.append(nongp_row["sentence"].strip())

            add(sentence, prefix, unambs, "amouyal", gp_type)

    print(f"  Amouyal: done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def deduplicate(items):
    seen = set()
    out = []
    for item in items:
        key = item["sentence"].lower().strip().rstrip(".")
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


if __name__ == "__main__":
    print("Processing datasets...")
    process_sap()
    process_syntaxgym()
    process_wang()
    process_jurayj()
    process_amouyal()

    entries_deduped = deduplicate(entries)
    print(f"\nTotal entries: {len(entries)} -> {len(entries_deduped)} after dedup")

    by_source = Counter(e["source"] for e in entries_deduped)
    by_type = Counter(e["type"] for e in entries_deduped)
    has_unamb = Counter(e["type"] for e in entries_deduped if e["unambiguous"])

    print("\nBy source:")
    for s, n in sorted(by_source.items()):
        print(f"  {s}: {n}")
    print("\nBy type:")
    for t, n in sorted(by_type.items()):
        u = has_unamb.get(t, 0)
        print(f"  {t}: {n}  ({u} with unambiguous variants)")

    with open(OUT, "w") as f:
        json.dump(entries_deduped, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}")
