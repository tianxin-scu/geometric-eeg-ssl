# Presentation Slides — Geometrically Inductive Self-Supervised Learning for EEG

**Constraint:** 5-minute talk per student, 6–7 slides max, course requirement is "as many figures as possible, less text." I use 6 slides plus a Title slide and a References slide bookending — 7 content frames total — and target ~45 s per slide.

The slide bodies below are bullet outlines plus a "Figure" placeholder describing exactly what visual goes in the slot. I deliberately keep the bullet count to ≤4 short lines per slide.

---

## Slide 1 — Title

> **Geometrically Inductive Self-Supervised Learning for EEG**
> *A Design Study of Spatial Attention Across Sessions, Subjects, and Montages*
>
> Tianxin Zhou — ECEN 525, Brain-Computer Interaction — Spring 2026
>
> github.com/tianxin-scu/geometric-eeg-ssl

**Figure:** none needed; centred title only.

---

## Slide 2 — Problem statement

**Title:** EEG is non-stationary by default — and current foundation models bake geometry into parameters.

**Bullets (≤3 lines):**
- EEG drifts on 3 axes: **session**, **subject**, **montage** — all are geometric.
- Codex models (LaBraM, EEGPT) store `embed[C3]`: identity-indexed, fails on unseen layouts.
- Channel-independent models (BIOT) drop spatial structure entirely.

**Figure (large, ~70 % of slide):**
Three-panel cartoon of a head with electrodes. Panel 1: same subject, day 1 vs day 2, electrodes shifted 5 mm (session). Panel 2: small head vs large head, same electrode names but different positions (subject). Panel 3: 64-channel cap → 3-channel headset, completely different layout (montage). One callout label across all three: *"The geometry has changed — codex parameters cannot."*

---

## Slide 3 — Solution: geometry-conditioned attention

**Title:** Replace per-electrode codex with a shared function over $g_{ij}$.

**Bullets (≤3 lines):**
- $g_{ij} = [\,\|p_i - p_j\|,\;p_i - p_j\,] \in \mathbb{R}^4$ — distance + signed 3-D displacement.
- Spatial attention is a shared MLP over $g_{ij}$, **no per-electrode parameters**.
- Three variants: **G1** (score-only) · **G2** (value-only) · **G3** (both).

**Figure (side-by-side, dominates slide):**
Two-column diagram (reuse `fig_inductive.tex` content from proposal):
- **Left — Transductive (codex):** `embed[C3] + patch → standard QK^T attention`. Box highlighted "fails to transfer."
- **Right — Geometric:** `patch + g_{ij} → score/value MLP → attention`. Box highlighted "transfers to any montage."

Add a tiny ribbon at the bottom showing the full training pipeline as a single horizontal flow: `raw EEG → patch → spatial attn → temporal → momentum target`, with a dashed EMA arrow. (Mini version of `fig_architecture.tex`.)

---

## Slide 4 — Results: in-distribution + ablation

**Title:** In-distribution: geometric = codex > channel-independent. Within geometric: G3 wins.

**Bullets (≤3 lines):**
- E1 (PhysioNet MI LOSO): G1 = 0.342, Codex = 0.341, Chind = 0.333.
- E2b paired Wilcoxon (G1 vs codex, n=105): **p = 0.911** — not significant.
- E5 ablation: **G3** best on both PhysioNet MI (0.350) and BCIC-2B (0.532).

**Figures (two small bar charts side-by-side):**

*Chart A — E1 bar chart on PhysioNet MI:*
```
Chind      ███████████████████ 0.333
Codex      ████████████████████ 0.341
G1         ████████████████████ 0.342
```
Y-axis label: "LOSO BAC (n=105)". Chance line at 0.25.

*Chart B — E5 G1/G2/G3 grouped bars on PhysioNet MI + BCIC-2B:*
```
              PhysioNet MI    BCIC-2B
G1            0.342           0.509
G2            0.343           0.509
G3 (winner)   0.350           0.532   ← highlight
```

---

## Slide 5 — Results: cross-montage transfer (the headline)

**Title:** Cross-montage transfer (64 ch → 3 ch): codex-nn matches the geometric encoder.

**Bullets (≤3 lines):**
- Pretrain PhysioNet MI 64-channel, **zero-shot** evaluate on BCIC-2B 3-channel.
- Codex with nearest-neighbour fallback **slightly beats** geometric G1.
- The nn fallback exploits geometry *at transfer time* — recovers the inductive advantage.

**Figure (single bar chart, large):**
```
G1 (geometric)         ██████████████████ 0.509 ± 0.028
Codex + random         ██████████████████ 0.513 ± 0.024
Codex + nearest-nbr    ███████████████████ 0.525 ± 0.022  ← strongest baseline
```
Y-axis: "BCIC-2B LOSO BAC (n=9, zero-shot)". Chance line at 0.50.
Annotate the +0.016 BAC gap between G1 and codex-nn.

---

## Slide 6 — Conclusion + future work

**Title:** Honest reading: a design study with a sharper target.

**Bullets (≤4 lines):**
- Spatial structure helps; **geometric vs codex is a tie** at pilot scale.
- **G3** is the strongest geometric configuration.
- **Codex-nn** is a stronger transfer baseline than the proposal anticipated.
- Next: **(a)** richer descriptors w/ G3, **(b)** larger pretraining, **(c)** high-density montages where nn breaks down.

**Figure (right half of slide, small):**
A small "result vs hypothesis" matrix:

|                        | Predicted     | Observed     |
| ---------------------- | ------------- | ------------ |
| In-distribution        | tie           | **tie ✓**    |
| Cross-subject LOSO     | geometric +   | tie          |
| Cross-montage transfer | geometric ++  | codex-nn ≈ + |

Plus: GitHub QR code in bottom-right of the slide pointing to the repo.

---

## Slide 7 — References (key citations only)

**Bullets (logos / textual list, ≤6 lines):**
- EEGPT — Wang et al., NeurIPS 2024 (training recipe)
- BYOL — Grill et al., NeurIPS 2020 (momentum alignment)
- MAE — He et al., CVPR 2022 (masked reconstruction)
- GraphSAGE — Hamilton et al., NeurIPS 2017 (inductive vs transductive vocabulary)
- GAT — Veličković et al., ICLR 2018 (attention over relational features)
- LaBraM — Jiang et al., ICLR 2024; BIOT — Yang et al., NeurIPS 2023 (codex / channel-independent baselines)

Footer: "Full reference list and per-experiment numbers in the project report."

---

## Talk track (5 min budget, ~45 s per slide)

1. **Title (~15 s).** Identify project + that this is a design study.
2. **Problem (~45 s).** EEG drifts on three axes; both camps of foundation models fail to handle the geometric component. Point at the three-panel cartoon.
3. **Solution (~50 s).** Define $g_{ij}$; explain G1/G2/G3 in one sentence each; show the two-column inductive diagram.
4. **In-dist + ablation (~50 s).** State that in-distribution is a tie; Wilcoxon p=0.911 confirms it; G3 wins among geometric.
5. **Cross-montage (~50 s).** Walk through the bar chart; emphasize the codex-nn fallback as the surprise.
6. **Conclusion (~55 s).** Read the result honestly: tie at pilot scale, G3 is the strongest geometric configuration, codex-nn is a tougher baseline than expected; three specific follow-ups.
7. **References (~10 s).** Briefly acknowledge primary citations, end on "report has the full details."

Total: ~5 min including the few seconds of transitions.
