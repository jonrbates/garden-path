# Garden-Path Sentences in Language Models

Mechanistic experiments on how transformer language models process garden-path sentences — grammatical sentences that lead the reader (or model) to an incorrect parse before disambiguating.

Blog post: [Unexpecting the expected in language models](https://jonrbates.ai/2025/07/26/garden-path-sentences.html)

## Key findings

Early MLP layers build strong expectations for natural (dominant-parse) continuations. Ablating MLP L0 increases control surprisal by +8–10 bits in Llama and Mistral, confirming these layers are critical for predicting expected continuations. However, garden-path surprisal is unchanged by the same intervention — the model's surprise at the disambiguating token does not depend on the early-layer commitment.

Three architectural patterns emerge across model families:
- **Llama / Mistral**: Early MLPs (L0, L1) concentrate natural-parse prediction
- **Gemma-2**: Early attention layers (attn L2) carry the selective effect instead
- **Qwen**: Weak garden-path effect at baseline across model sizes (7B, 14B)

See [FINDINGS.md](FINDINGS.md) for full results.

## Data

**`data/data.json`** — Original stimulus set (n=53, 33 with controls). Used in Experiments 1–2.

**`data/derived.json`** — Expanded stimulus set (n=158, all with controls). Used in Experiment 3. Compiled deterministically by `compile_derived.py`.

**`data/compiled.json`** — Intermediate compilation from raw benchmark files. Produced by `compile_datasets.py`.

### Raw benchmark data

Raw benchmark data is not included. To reproduce `compiled.json` from source, download and place in `data/`:

| Directory | Source | URL |
|-----------|--------|-----|
| `data/sap_benchmark/` | Stachenfeld et al., SAP Benchmark | https://github.com/google-deepmind/sap_benchmark |
| `data/syntaxgym/` | Hu et al., SyntaxGym | https://syntaxgym.org |
| `data/wang_sf/` | Wang & Sturt | https://github.com/caplabnyu/sapbenchmark |

Then run `python compile_datasets.py` followed by `python compile_derived.py`.

## Experiments

| Script | Experiment | Description |
|--------|------------|-------------|
| `compute_layer_ablation.py` | 1–2 | Per-layer ablation across all layers, n=33, 4-bit |
| `compute_selectivity.py` | 3 | Early-layer ablation (L0–L2), n=157, bf16, 4 model families |
| `compute_derived_surprisal.py` | 3 (baseline) | Juke-point surprisal for derived.json |
| `compute_prepend.py` | Supplementary | Does cueing "garden path sentence" reduce surprisal? |
| `compute_selfprime.py` | Supplementary | Does repeating the sentence reduce surprisal? |
| `compute_surprise.py` | Baseline | Per-token surprisal and top-k analysis |

## Models

Experiments 1–2 (4-bit): Llama-3.2-3B, Llama-3.1-8B, Qwen-2.5-14B, Llama-3.1-70B

Experiment 3 (bf16): Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B, Qwen-2.5-7B

## Infrastructure

- `gps.py` — Model wrapper with ablation, patching, and surprisal measurement
- `theme.py` — Matplotlib styling
- `plot_layer_ablation.py`, `plot_metrics.py`, `plot_parse_tree.py` — Figures
- `verify_tokenization.py` — Validates juke-point tokenization consistency across models

## Requirements

PyTorch, Transformers, NumPy, SciPy. See `uv.lock` for pinned versions.
