"""
Demo an instruction tuned chat LLM.
"""

import json
import os
import pandas as pd

from colorama import Fore, Style, init
from gps import Model
from huggingface_hub import login
from typing import List, Sequence


def _render_token(piece: str) -> str:
    """Render a single token with correct spacing for BPE (Ġ) or SentencePiece (▁)."""
    if piece in {"<s>", "</s>", "<|eot_id|>", "<|eos_token|>"}:
        return ""
    if piece == "Ċ":  # GPT-2 newline token
        return "\n"
    if piece.startswith("Ġ") or piece.startswith("▁"):
        return " " + piece[1:]
    return piece


def topk_surprisal_indices(
    stats: Sequence[dict], k: int = 2, skip_first_n: int = 2
) -> List[int]:
    """Return indices of the top-k highest surprisal tokens, skipping the first n positions."""
    vals = [r["surprisal"] for r in stats]
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    filtered = [i for i in order if i >= skip_first_n]
    return filtered[:k]


def format_surprisal_tokens(
    stats: Sequence[dict],
    *args,
    prefix_length: int,
    skip_first_n: int,
) -> str:
    """Return a string with tokens, coloring the two most surprising (after skipping first N)."""
    if len(stats) <= skip_first_n:
        return "Not enough tokens."

    topk = topk_surprisal_indices(stats, k=2, skip_first_n=skip_first_n)
    if not topk:
        return f"No eligible tokens after position {skip_first_n - 1}."

    out: List[str] = []
    for i, s in enumerate(stats):
        disp = _render_token(s["next_token"])
        if not disp:
            continue
        if i == topk[0]:
            disp = (
                Fore.RED
                + Style.BRIGHT
                + disp
                + f"[{s['surprisal']:.2f}]"
                + Style.RESET_ALL
            )
        elif len(topk) > 1 and i == topk[1]:
            disp = (
                Fore.YELLOW
                + Style.BRIGHT
                + disp
                + f"[{s['surprisal']:.2f}]"
                + Style.RESET_ALL
            )
        if i == prefix_length - 1:
            disp = Style.BRIGHT + "/&/" + disp + Style.RESET_ALL
        out.append(disp)
    return "".join(out)


def get_depth(stats: Sequence[dict], *, prefix_length: int, skip_first_n: int) -> int:
    """Score agreement between annotator prefix and surprisal rank."""
    topk = topk_surprisal_indices(stats, k=len(stats), skip_first_n=skip_first_n)
    if not topk:
        return float("inf")
    return next((i for i, v in enumerate(topk) if v == prefix_length), float("inf"))


if __name__ == "__main__":
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    init(autoreset=True)
    # "TheBloke/Llama-2-7B-Chat-GPTQ", "Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it"
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    e = Model(MODEL_ID)

    print("Quick example:")
    sentence = "The old man"
    inputs = e.tokenizer(sentence, return_tensors="pt").to(e.model.device)
    out = e.model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=False)
    print(e.tokenizer.decode(out[0], skip_special_tokens=True))

    # --------------------------------------------------------------------------- #
    #  demo                                                                       #
    # --------------------------------------------------------------------------- #

    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "data/data.json")
    df = pd.read_json(file_path)

    cache_path = os.path.join(base_path, "cache", "compute_surprise_cache.json")

    results = {}
    try:
        with open(cache_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        with open(cache_path, "w") as f:
            json.dump({}, f)

    exp_key = "baseline"
    print(f"Experiment: {exp_key}")
    for row_idx, row in df.iterrows():
        sentence, prefix = row["sentence"], row["prefix"]
        # skip if cached
        if sentence in results and exp_key in results[sentence]:
            print(f"Skipping cached result for {sentence} with {exp_key}")
            continue
        stats = e.compute_sentence_metrics(sentence, k=10)
        prefix_length = e.tokenizer(prefix, return_tensors="pt").input_ids.size(-1)
        results.setdefault(sentence, {}).setdefault(exp_key, {})
        results[sentence][exp_key]["string"] = format_surprisal_tokens(
            stats, prefix_length=prefix_length, skip_first_n=2
        )
        results[sentence][exp_key]["prefix length"] = prefix_length
        results[sentence][exp_key]["depth"] = get_depth(
            stats, prefix_length=prefix_length, skip_first_n=2
        )
        results[sentence][exp_key]["surprisals"] = [r["surprisal"] for r in stats]
        results[sentence][exp_key]["entropies"] = [r["entropy"] for r in stats]
        results[sentence][exp_key]["juke_surprisal"] = (
            stats[prefix_length - 1]["surprisal"]
            if prefix_length - 1 < len(stats)
            else None
        )
        results[sentence][exp_key]["stats"] = stats
        s = results[sentence][exp_key]["string"]
        depth = results[sentence][exp_key]["depth"]
        print(f"Token String: {s}")
        print(f"Depth: {depth}")
        print("-" * 40)

    # cache results
    with open(cache_path, "w") as f:
        json.dump(results, f)
