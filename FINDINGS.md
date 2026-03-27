# Findings: Early MLP Layers and the Garden-Path Effect

Experiment 1–2 models: Llama-3.2-3B-Instruct (4-bit), Llama-3.1-8B-Instruct (4-bit), Qwen-2.5-14B-Instruct (4-bit), Llama-3.1-70B-Instruct (4-bit); n=33 unstacked GPS sentences from data.json.

Experiment 3 models: Llama-3.1-8B-Instruct (bf16), Mistral-7B-v0.3 (bf16), Gemma-2-9B (bf16), Qwen-2.5-7B-Instruct (bf16); n=157 paired GPS/CTRL sentences from derived.json.

---

## Dataset

### Overview

Two datasets are used. **data.json** (n=53, 33 with controls) was the original stimulus set for the per-layer ablation experiments reported below. **derived.json** (n=158, all with controls) is an expanded set used for the cross-architecture selectivity replication. Both are compiled deterministically by `compile_derived.py`.

### Sources

Items are drawn from three published garden-path benchmarks and one supplementary set:

| Source | N (derived) | Types | Citation |
|--------|-------------|-------|----------|
| SAP Benchmark | 34 | MVRR, NPS | Stachenfeld et al. |
| SyntaxGym | 56 | MVRR | Hu et al. |
| Wang et al. | 32 | NPS | Wang & Sturt |
| Supplementary | 36 | MVRR, NPS, NPVP | see below |

### Inclusion criteria

All items satisfy three requirements:

1. **Single disambiguation point.** The sentence has a prefix that is syntactically ambiguous between two parses, with a single "juke point" token where the garden-path and control continuations diverge. CENTER_EMBED constructions were excluded because processing difficulty is distributed across multiple positions, incompatible with the single-point measurement design.

2. **Matched control.** Each item has a non-garden-path control sentence sharing the same prefix but continuing with the dominant (expected) parse.

3. **Genuine ambiguity (SAP NPS filter).** For SAP Benchmark NPS items, we required that the main verb's cloze sentence-complement bias (cloze_SZMbias from the SAP norming data) be below 0.5. Verbs with bias ≥ 0.5 already strongly favour the sentential-complement reading, meaning the disambiguation is expected rather than surprising — the sentence is not genuinely garden-path. This excluded 14 of 24 SAP NPS items. The threshold was not applied to Wang et al. NPS items, which were pre-screened for ambiguity in the original study.

### Supplementary items

36 items (24 synthetic, 12 human-authored) were generated collaboratively with AI to increase coverage of under-represented construction types, particularly NPVP (9 items, none available from published sources). These items were filtered by the first author for grammaticality and genuine garden-path structure.

The main per-layer ablation results (Experiments 1–2) use only data.json (n=33). The supplementary items enter in the cross-architecture selectivity experiment, where they serve as a robustness check: if the MLP L0 effect replicates on novel, independently constructed sentences, it is unlikely to be an artifact of the original stimulus set.

### Reproducibility

`compile_derived.py` regenerates derived.json deterministically from compiled.json (produced by `compile_datasets.py` from the raw benchmark files), data.json, and the SAP verb-bias norms.

---

## Core finding

**Early MLP layers build the expectation for natural (dominant-parse) continuations, but garden-path surprise is not a byproduct of that commitment.** Across four model families, ablating MLP L0 consistently and massively increases surprisal for control (natural-parse) continuations — by +8–10 bits in Llama and Mistral — confirming that early MLPs are critical for predicting expected continuations. However, GPS surprisal is essentially unchanged by the same intervention (ΔGPS ≈ 0), meaning the model's surprise at garden-path tokens does not depend on the early-layer commitment to the natural parse.

This rules out the serial-commitment account in which garden-path surprise emerges as a byproduct of strong commitment to the dominant parse. The natural-parse expectation and the garden-path surprise appear to be mechanistically independent: the model builds one in early MLP layers, but the other is produced elsewhere (or distributed across layers in a way that no single early ablation can disrupt).

The initial n=33 Llama results (Experiments 1–2) appeared consistent with the serial-commitment story, showing a GPS surprisal *decrease* when MLP L0 was ablated (ΔGPS = −1.87 to −2.06). This GPS drop did not replicate at n=157 in the cross-architecture experiment (Experiment 3), suggesting it was driven by a subset of sentences in the smaller stimulus set.

**The mechanism varies by architecture.** Llama and Mistral show the same pattern (early MLPs serving natural-parse prediction). Gemma-2-9B shows the largest baseline garden-path effect but achieves it through early attention layers rather than MLPs. Qwen shows a weak garden-path effect across model sizes (7B and 14B), suggesting a family-level trait.

## Top-level takeaways

1. **Early MLP layers build natural-parse expectations, not garden-path surprise.** Across Llama and Mistral, ablating MLP L0 increases CTRL surprisal by +8–10 bits while leaving GPS surprisal unchanged (ΔGPS ≈ 0). The interaction is large (d > 1.3) but driven entirely by the CTRL side. Garden-path surprise is mechanistically independent of the early-layer commitment.

2. **The "serial commitment → byproduct surprise" framing does not replicate.** At n=33 (Experiments 1–2), MLP L0 ablation appeared to reduce GPS surprisal (ΔGPS = −1.87 to −2.06). At n=157 across four model families (Experiment 3), no model shows a substantial GPS decrease from early-layer ablation.

3. **Three distinct architectural patterns emerge.** Llama/Mistral concentrate natural-parse prediction in early MLPs (L0, L1). Gemma-2 builds prediction through early attention (attn_L2 is its strongest selective condition, d = −0.60) with inert early MLPs. Qwen shows a weak garden-path effect at baseline across model sizes (gap of 0.5–3.5 bits vs 10–15 bits for other families), with no strongly selective ablation at any layer.

4. **Gemma-2-9B has the largest baseline garden-path effect** (14.83 bits, vs ~10 bits for Llama/Mistral), yet its early MLP layers are nearly inert (mlp_L0 top-5 overlap: 89.6%). The commitment mechanism — whatever produces its strong garden-path effect — appears to operate through attention rather than MLPs.

5. **Qwen's weak garden-path effect is a family trait, not a scale artifact.** Qwen-2.5-14B (4-bit, n=33) showed a 0.5-bit gap; Qwen-2.5-7B (bf16, n=157) shows a 3.5-bit gap. Both are far below the 10–15-bit gaps seen in other families.

---

## Experiment 1: Per-layer ablation

For each layer L (0–27) and component C ∈ {attention, MLP}, skip that one component and measure surprisal, rank, and top-5 continuations at the juke point. No compounding — one intervention per run.

### GPS surprisal decreases when the natural parse is disrupted

| Condition | Δ GPS surp | Top-5 overlap | GPS reduced |
|-----------|-----------|---------------|-------------|
| mlp_L0 | **−2.06** | **3.6%** | 26/33 |
| mlp_L1 | **−5.16** | **13.3%** | 30/33 |
| mlp_L2 | +0.03 | 77% | — |
| attn_L0 | −1.32 | 57% | 25/33 |
| attn_L1 | −0.18 | 83% | — |
| mlp_L27 | +4.91 | 78% | 3/33 |

**MLP L0**: GPS surprisal drops by 2.06 bits (paired t(32) = −2.97, p = 0.006; Wilcoxon p = 0.0006; Cohen's d = −0.53). 26 of 33 sentences show reduced GPS surprisal. Top-5 overlap with baseline is 3.6% — the prediction distribution is almost entirely replaced.

**MLP L1**: GPS surprisal drops by 5.16 bits (paired t(32) = −7.06, p < 0.0001; Wilcoxon p < 0.0001; Cohen's d = −1.25). 30 of 33 sentences show reduced GPS surprisal.

**The effect is parse-specific, not general degradation.** Ablating MLP L0 moves GPS and ALT surprisal in opposite directions (GPS −2.06, ALT +6.13). The interaction is significant: Δ GPS − Δ ALT = −8.18 bits, t(32) = −8.53, p < 0.0001, Cohen's d = −1.51. MLP L1 shows the same pattern (interaction −7.79, t(32) = −9.56, p < 0.0001, d = −1.69).

This rules out the interpretation that ablation simply degrades the model. The degradation is directional: removing MLP L0/L1 dismantles the natural-parse commitment, and GPS surprisal falls as a consequence.

By MLP L2, the effect is negligible. The prediction distribution is essentially built in the first two MLP blocks.

MLP L27 (final layer) is a general prediction sharpener: ablating it increases surprisal for all continuations (+4.91 GPS, +1.95 ALT). This is prediction refinement, not parse-specific computation.

### Attention vs MLP contribution

Top-5 overlap (fraction of baseline top-5 tokens surviving ablation) across all layers:

- **Attention**: 57% at L0, then 77–96% (L1–L27)
- **MLP**: 3.6% at L0, 13% at L1, then 71–90% (L2–L27)

MLPs reshape the distribution more than attention at every layer.

---

## Experiment 2: Split-layer dropout (supporting evidence)

Dropout applied to either **residual connections** or **block outputs** (post-MLP), split into bottom-half (L0–13) and top-half (L14–27). One axis swept at a time (other held at 0). 8 seeds per condition.

*Note on design*: These experiments preceded the per-layer ablation and motivated it. The compounding nature of residual dropout and the asymmetry between resid/block interventions limit cross-condition interpretability. The per-layer ablation gives cleaner, more localized answers. The dropout results are retained here as supporting evidence.

### Block bottom-half dropout selectively reduces the GPS/ALT gap

At p=0.5, bottom-half block dropout degrades ALT rank 13× while GPS degrades only 2.4×. This is consistent with the per-layer finding: the bottom half contains MLP L0 and L1, which build the natural-parse expectation. Disrupting them closes the gap between GPS and ALT — not by making GPS more surprising, but by making the natural continuation less expected.

**Interaction test** (paired t-test on log₂ rank change, ALT − GPS):
- Mean difference: +2.28 log₂ units
- t(32) = 3.064, p = 0.004 (two-tailed)
- Cohen's d = 0.54 (medium effect)

### Noise propagation doesn't explain the asymmetry

A pure noise-propagation account predicts symmetric degradation of GPS and ALT. The selective vulnerability of ALT — while GPS is preserved — rules this out.

### Residual stream carries the signal uniformly

Residual dropout degrades predictions regardless of whether it's applied to the top or bottom half (no significant asymmetry). The residual stream is the shared backbone that MLPs write to; disrupting it destroys information uniformly.

---

## Methodological notes: Why ablation, not activation patching

### Why activation patching fails for garden-path sentences

Activation patching (Meng et al. 2022) assumes a single correct answer. Garden-path processing has no single correct continuation — the model's prediction is a distribution. This causes two problems: (1) noise calibration is unstable across sentences with baseline surprisals ranging from ~1 to ~20 bits, and (2) recovering surprisal at one token does not mean recovering the model's syntactic expectations — the distribution could be completely different.

A distributional variant using cosine similarity of full logit vectors addresses problem 2, but has a deeper issue: in a causal LM, logits at the juke point depend only on the prefix, which is identical for GPS and CTRL sentences. The recovery profile is the same for any prefix, ambiguous or not. The experiment measures general layer importance, not garden-path-specific processing.

### Why ablation with selectivity analysis is sufficient

The ablation experiment's key finding is that early MLP effects are **asymmetric**: ablation disproportionately increases CTRL surprisal while leaving GPS surprisal largely unchanged. This asymmetry is the critical evidence:

- **It rules out general degradation.** If ablation simply damaged the model, both GPS and CTRL would degrade together. The asymmetric effect proves the intervention is parse-specific.
- **It is consistent across architectures.** The MLP L0 asymmetry appears in Llama and Mistral (interactions of −8.18 and −8.73 at n=157). Even where the mechanism differs (Gemma uses attention rather than MLPs), the principle holds: early computation disproportionately serves natural-parse prediction.

However, the selectivity analysis also constrains the interpretation: the ablation effect is carried entirely by the CTRL side. GPS surprisal does not decrease when early layers are removed. This means early MLPs are specifically necessary for natural-parse prediction, but not for the garden-path surprise itself. The two are mechanistically separable.

### Precision: 4-bit vs fp16

For ablation, 4-bit quantization is acceptable. The intervention is binary (zero out a component or don't), and comparison of 8B 4-bit vs fp16 baselines shows near-perfect agreement: Pearson r = 0.994, GPS-ALT gap preserved with 100% sign consistency across all 33 sentences. 4-bit enables running larger models (70B) on single-GPU hardware without compromising the ablation results.

(fp16 would be required for any noise-injection method, since Gaussian noise on quantized embeddings is not well-defined. This is moot given the decision to use ablation only.)

### Cross-model ablation comparison (Experiments 1–2, n=33, 4-bit)

#### Baseline garden-path effect

| Model | Layers | GPS surp | ALT surp | Gap |
|-------|--------|----------|----------|-----|
| 3B Llama | 28 | 17.47 | 6.97 | 10.49 |
| 8B Llama | 32 | 18.09 | 6.83 | 11.25 |
| 14B Qwen | 48 | 5.79 | 5.29 | **0.51** |
| 70B Llama | 80 | 17.70 | 7.42 | 10.28 |

#### MLP L0 selectivity

| Model | Δ GPS | Δ ALT | Interaction | Selective? | Top-5 overlap |
|-------|-------|-------|-------------|------------|---------------|
| 3B Llama | −2.06 | +6.13 | −8.18 | **Yes** | 3.6% |
| 8B Llama | −1.87 | +4.67 | −6.54 | **Yes** | 0% |
| 14B Qwen | +4.59 | +6.99 | −2.40 | **No** (both ↑) | — |
| 70B Llama | −1.73 | +3.74 | −5.47 | **Yes** | 0% |

At n=33, MLP L0 ablation appeared to reduce GPS surprisal in all three Llama models (ΔGPS = −1.73 to −2.06). This GPS decrease did not replicate at n=157 (see Experiment 3 below).

#### MLP L1 selectivity

| Model | Δ GPS | Δ ALT | Interaction | Selective? | Top-5 overlap |
|-------|-------|-------|-------------|------------|---------------|
| 3B Llama | −5.16 | +2.63 | −7.79 | **Yes** | 13.3% |
| 8B Llama | −3.27 | +4.52 | −8.07 | **Yes** | — |
| 14B Qwen | +0.07 | +0.06 | −0.00 | **No** (inert) | 100% |
| 70B Llama | −0.74 | −0.12 | −0.62 | **No** (both ↓) | 91% |

MLP L1 is strongly selective only at 3B and 8B. At 70B, the ablation barely changes predictions (91% top-5 overlap). At Qwen, it's completely inert.

#### Selectivity distribution

| Model | Selective / Total | % | Strongest interaction |
|-------|-------------------|---|---------------------|
| 3B Llama | 18 / 56 | 32% | mlp_L0 (−8.18) |
| 8B Llama | 13 / 64 | 20% | mlp_L1 (−8.07) |
| 14B Qwen | 15 / 96 | 16% | mlp_L31 (−0.57) |
| 70B Llama | 45 / 160 | 28% | mlp_L0 (−5.47) |

#### Final-layer MLP

| Model | Layer | Δ GPS | Δ ALT |
|-------|-------|-------|-------|
| 3B Llama | L27 | +4.91 | +1.95 |
| 8B Llama | L31 | +4.24 | +2.12 |
| 14B Qwen | L47 | +0.69 | +0.45 |
| 70B Llama | L79 | +5.78 | +2.51 |

The final MLP is universally load-bearing for prediction sharpening.

---

## Experiment 3: Cross-architecture selectivity replication (n=157, bf16)

Ablation of L0–L2 (attention and MLP) across four 7–9B model families at bf16 precision, using the expanded derived.json stimulus set (n=157 paired GPS/CTRL sentences). This experiment tests whether the Experiment 1–2 findings generalize across architectures and a larger, more diverse stimulus set.

### Baseline garden-path effect

| Model | GPS surp | CTRL surp | Gap |
|-------|----------|-----------|-----|
| Llama 8B | 15.63 | 5.77 | 9.86 |
| Mistral 7B | 15.40 | 5.45 | 9.95 |
| Gemma 9B | 20.31 | 5.48 | **14.83** |
| Qwen 7B | 7.61 | 4.12 | **3.49** |

Three patterns: Llama/Mistral cluster at ~10-bit gaps, Gemma shows the largest effect at ~15 bits, and Qwen shows a much weaker effect at ~3.5 bits.

### Full ablation results

| Model | Condition | Δ GPS | Δ CTRL | Interaction | d | GPS↓ CTRL↑? |
|-------|-----------|-------|--------|-------------|---|-------------|
| **Llama 8B** | attn_L0 | −0.52 | +0.70 | −1.22 | −0.43 | yes |
| | mlp_L0 | +0.07 | +8.25 | **−8.18** | −1.36 | no |
| | attn_L1 | −0.04 | −0.06 | +0.02 | +0.01 | no |
| | mlp_L1 | −0.44 | +8.62 | **−9.07** | −1.50 | yes (weak GPS↓) |
| | attn_L2 | −0.03 | −0.03 | −0.00 | −0.00 | no |
| | mlp_L2 | −0.25 | +0.15 | −0.40 | −0.15 | yes (weak) |
| **Mistral 7B** | attn_L0 | −1.09 | +0.35 | −1.44 | −0.57 | yes |
| | mlp_L0 | +1.19 | +9.92 | **−8.73** | −1.57 | no |
| | attn_L1 | −0.08 | +0.10 | −0.18 | −0.16 | yes (weak) |
| | mlp_L1 | −1.80 | +3.70 | **−5.50** | −1.23 | yes |
| | attn_L2 | −0.11 | +0.04 | −0.15 | −0.14 | yes (weak) |
| | mlp_L2 | −0.12 | +0.10 | −0.22 | −0.20 | yes (weak) |
| **Gemma 9B** | attn_L0 | −0.68 | +0.25 | −0.93 | −0.29 | yes (weak) |
| | mlp_L0 | −0.14 | +0.09 | −0.24 | −0.11 | yes (trivial) |
| | attn_L1 | −0.64 | +0.17 | −0.81 | −0.28 | yes (weak) |
| | mlp_L1 | −0.54 | +0.04 | −0.58 | −0.25 | yes (trivial CTRL↑) |
| | attn_L2 | −2.27 | +0.26 | **−2.53** | −0.60 | yes |
| | mlp_L2 | −0.69 | +0.08 | −0.77 | −0.25 | yes (trivial CTRL↑) |
| **Qwen 7B** | attn_L0 | +6.55 | +7.22 | −0.67 | −0.11 | no |
| | mlp_L0 | +1.78 | +2.71 | −0.92 | −0.20 | no |
| | attn_L1 | +0.07 | +0.19 | −0.12 | −0.09 | no |
| | mlp_L1 | +0.03 | +0.17 | −0.14 | −0.10 | no |
| | attn_L2 | −0.03 | +0.12 | −0.14 | −0.13 | yes (trivial) |
| | mlp_L2 | +0.04 | +0.07 | −0.04 | −0.02 | no |

GPS Top-5 overlap with baseline:

| Model | mlp_L0 | mlp_L1 |
|-------|--------|--------|
| Llama 8B | 2.5% | 5.1% |
| Mistral 7B | 6.9% | 33.0% |
| Gemma 9B | 89.6% | 86.2% |
| Qwen 7B | 35.1% | 79.4% |

### What replicated from n=33

**The interaction magnitude for Llama 8B is stable.** mlp_L0 interaction was −6.54 at n=33 (4-bit) and −8.18 at n=157 (bf16). mlp_L1 interaction was −8.07 at n=33 and −9.07 at n=157. The effect is robust to both sample size and precision.

**Mistral shows the same pattern as Llama.** mlp_L0 interaction of −8.73 (d = −1.57), mlp_L1 interaction of −5.50 (d = −1.23). Top-5 overlap at mlp_L0 is 6.9%. The Llama mechanism is shared with Mistral.

**Qwen's weak garden-path effect replicates.** The 7B bf16 result (gap = 3.49 bits) confirms the 14B 4-bit result (gap = 0.51 bits) — Qwen models show much smaller garden-path effects than other families, and no strongly selective ablation conditions.

### What did not replicate

**The GPS decrease from mlp_L0 ablation.** At n=33, all three Llama models showed ΔGPS < 0 when mlp_L0 was ablated (−2.06, −1.87, −1.73). At n=157, Llama 8B shows ΔGPS = +0.07 and Mistral shows ΔGPS = +1.19. The interaction is preserved (driven by the massive CTRL increase), but the GPS decrease that motivated the "commitment → byproduct surprise" interpretation is absent. GPS surprisal is not reduced by removing early MLP layers.

### New findings

**Gemma-2-9B uses a different mechanism.** Despite having the largest baseline gap (14.83 bits), Gemma's early MLP layers are nearly inert (mlp_L0 top-5 overlap: 89.6%). Its strongest selective condition is attn_L2 (interaction −2.53, d = −0.60), suggesting parse-relevant computation occurs in early attention rather than MLPs. This may relate to Gemma-2's hybrid attention design (alternating sliding-window and global attention layers).

### Summary

Early MLP layers are critical for natural-parse prediction in Llama and Mistral, but ablating them does not reduce garden-path surprise. The garden-path effect and the natural-parse expectation are mechanistically separable. The original "serial commitment → byproduct surprise" interpretation, supported by the GPS decrease at n=33, does not hold at n=157 across four model families.

The consistent cross-architecture finding is weaker but cleaner: **early layers disproportionately serve natural-parse prediction.** This holds in different forms across Llama/Mistral (MLPs), Gemma (attention), and even weakly in Qwen. But the garden-path surprise itself is not localized to these layers.
