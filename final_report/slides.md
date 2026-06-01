# Presentation Slides — Geometrically Inductive Self-Supervised Learning for EEG (v2)

**Constraint:** 5-minute talk per student, 6–7 slides max, course requirement is "as many figures as possible, less text." I use 5 content slides plus a Title slide and a References slide — 7 frames total — and target ~45 s per slide.

**v2 framing (changed from v1):** the codex baseline is deferred (it cannot attempt zero-shot cross-montage transfer by construction), so the comparison that matters is **geometric vs. channel-independent (chind)** — the only other encoder that *can* transfer to an unseen montage. The headline result is now a *positive*, significant one. The slide bodies below are bullet outlines (≤4 short lines) plus a "Figure" placeholder describing the visual.

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

**Title:** EEG is non-stationary by default — and the geometry is what changes.

**Bullets (≤3 lines):**
- EEG drifts on 3 axes: **session**, **subject**, **montage** — all geometric.
- Codex models (LaBraM, EEGPT) store `embed[C3]`: identity-indexed, so they **cannot run at all** on an unseen layout.
- Channel-independent models (BIOT) drop spatial structure entirely.

**Figure (large, ~70 % of slide):**
Three-panel cartoon of a head with electrodes. Panel 1: same subject, day 1 vs day 2, electrodes shifted ~5 mm (session). Panel 2: small head vs large head, same names, different positions (subject). Panel 3: 64-channel cap → 3-channel headset, completely different layout (montage). Callout across all three: *"The geometry has changed — codex parameters cannot."*

---

## Slide 3 — Solution: geometry-conditioned attention

**Title:** Replace the per-electrode codex with a shared function over $g_{ij}$.

**Bullets (≤4 lines):**
- $g_{ij} = [\,p_i,\; p_j,\; p_i - p_j,\; \|p_i - p_j\|\,] \in \mathbb{R}^{10}$ — absolute positions + signed displacement + distance.
- Spatial attention is a **shared MLP over $g_{ij}$**, with **no per-electrode parameters** → applies to any montage.
- Injection variant **G3**: $g_{ij}$ shapes both the attention *score-bias* and the *value*.
- The only rival that can also transfer is **chind** (no geometry); codex can't even be evaluated zero-shot.

**Figure (side-by-side, dominates slide):**
Two-column diagram (reuse `tex/fig_inductive.tex`):
- **Left — Transductive (codex):** `embed[C3] + patch → standard QKᵀ attention`. Box: "cannot transfer — no slot for unseen electrodes."
- **Right — Geometric:** `patch + g_{ij} → score/value MLP → attention`. Box: "transfers to any montage."

Bottom ribbon = the training pipeline as one horizontal flow: `raw EEG → patch → spatial attn → temporal → momentum target`, dashed EMA arrow (mini `tex/fig_architecture.tex`).

---

## Slide 4 — Experiment: leave-one-montage-out zero-shot transfer

**Title:** Train on a mix of montages, evaluate zero-shot on a montage never seen in pretraining.

**Bullets (≤4 lines):**
- Mixed-corpus SSL pretraining; hold out **one entire dataset**, freeze encoder, fit a linear probe on the held-out montage (genuinely zero-shot).
- Identical training for every variant — only the spatial encoder differs (fair comparison).
- Across all three held-out montages (balanced 9-subject corpus), **geometric ≥ chind on 3/3** — directionally consistent, though gaps are within noise at this small scale.

**Figure (small 3×3 table, directional):**
```
Held-out         geometric   pos_only    chind
Sleep-EDFx         0.593       0.597      0.589
BCIC-2B            0.509       0.508      0.504
PhysioNet MI       0.303       0.315      0.302
```
Caption: "Zero-shot BAC, 9-subject balanced corpus. Different tasks → different chance levels; read *within* each row (geometric vs chind), not across rows. Consistent direction, not yet significant — motivates scaling the one split we can scale."

---

## Slide 5 — Headline result: scale makes the geometric advantage significant

**Title:** With enough pretraining data, geometric attention beats chind on zero-shot transfer — significantly.

**Bullets (≤4 lines):**
- Scaled the **no-BCIC** split to a balanced **n=156** corpus (PhysioNet 78 + Sleep 78); zero-shot to BCIC-2B (BCIC's 9-subject ceiling is *why* this is the one feasible scaled split).
- **Geometric (full $g_{ij}$) is strongest: 0.545**, beating chind (0.509) by **+0.036**, **significant**: paired *t* p=0.023, Wilcoxon p=0.027, 7/9 subjects.
- The reduced **pos_only** descriptor *also* beats chind significantly: **+0.022**, p=0.013, 8/9 subjects → the signal is robust to dropping displacement/distance.
- Codex is absent because it cannot be evaluated here at all — that absence *is* the point.

**Figure (single bar chart, large):**
```
chind                  ████████████████ 0.509 ± 0.016
geometric (pos_only)   █████████████████ 0.531 ± 0.013   (p = 0.013 vs chind)
geometric (full g_ij)  ██████████████████ 0.545 ± 0.027  ← strongest (p = 0.023 vs chind)
```
Y-axis: "BCIC-2B zero-shot BAC (n=9 LOSO folds, n=156 pretrain)". Chance line at 0.50. Annotate both significant gaps over chind (full +0.036, pos_only +0.022).

---

## Slide 6 — Conclusion + honest limits + future work

**Title:** A bounded design study with a clean positive — and clear edges.

**Bullets (≤4 lines):**
- **Result:** geometry-conditioned attention transfers zero-shot to an unseen montage and beats the only comparable baseline (chind) significantly once given enough data.
- **Honest limits:** significant on **one** held-out montage (BCIC); the small-corpus tests are directional but underpowered (9-subject ceiling on the test set).
- **Why only one powered test:** BCIC's 9-subject cap forces it to be held out — the other two directions can't be both balanced *and* large with these three public datasets.
- **Future work:** a second distinct-montage held-out (e.g. high-density 128-ch); larger/diverse corpus; implement the codex baseline.

**Figure (right half, small):** result-vs-hypothesis matrix:

|                          | Predicted      | Observed                         |
| ------------------------ | -------------- | -------------------------------- |
| Direction (geom vs chind)| geometric +    | **geometric + (3/3 rows) ✓**     |
| Powered transfer test    | geometric ++   | **geometric ++, significant ✓**  |
| Generalizes across montages | hoped       | one montage powered; rest future |

Plus a GitHub QR code bottom-right pointing to the repo.

---

## Slide 7 — References (key citations only)

**Bullets (textual list, ≤6 lines):**
- EEGPT — Wang et al., NeurIPS 2024 (training recipe)
- BYOL — Grill et al., NeurIPS 2020 (momentum alignment)
- MAE — He et al., CVPR 2022 (masked reconstruction)
- GraphSAGE — Hamilton et al., NeurIPS 2017 (inductive vs transductive vocabulary)
- GAT — Veličković et al., ICLR 2018 (attention over relational features)
- LaBraM — Jiang et al., ICLR 2024; BIOT — Yang et al., NeurIPS 2023 (codex / channel-independent baselines)

Footer: "Full reference list and per-experiment numbers in the project report."

---

## Talk track (5 min budget, ~45 s per slide)

1. **Title (~15 s).** Identify project + that this is a one-quarter design study.
2. **Problem (~45 s).** EEG drifts on three geometric axes; codex models can't even run on an unseen montage, channel-independent models throw geometry away. Point at the three-panel cartoon.
3. **Solution (~50 s).** Define the 10-D $g_{ij}$; one shared MLP, no per-electrode parameters, G3 injection; note chind is the only fair rival because codex can't transfer.
4. **Experiment (~50 s).** Leave-one-montage-out, freeze, linear-probe zero-shot. At balanced 9-subject scale, geometric leads chind on all three held-out montages but within noise — so I scaled the one split I could.
5. **Headline (~55 s).** The n=156 no-BCIC run: geometric (full) is strongest at 0.545 and beats chind significantly (+0.036, p=0.023, 7/9); the pos_only descriptor also beats chind significantly (+0.022, p=0.013, 8/9). The codex column is empty *by construction* — it can't be evaluated.
6. **Conclusion (~55 s).** A clean positive on the powered test; honest that it's one montage and the small tests are underpowered by BCIC's 9-subject ceiling; three concrete follow-ups.
7. **References (~10 s).** Acknowledge primary citations; "report has the full details."

Total: ~5 min including transitions.
