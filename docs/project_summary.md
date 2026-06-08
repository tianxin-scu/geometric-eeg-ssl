# Project Summary — Geometrically Inductive SSL for EEG

Canonical current description, for report and talk drafting. Single author:
Tianxin Zhou. See [CLAUDE.md](../CLAUDE.md).

## 1. Idea

Replace the learned per-electrode **codex** of EEG foundation models with a
spatial-attention function conditioned on a **geometric descriptor** `g_ij`
computed from the 3-D scalp coordinates of each electrode pair. The function
holds no per-electrode parameters, so the same trained encoder applies to any
montage given coordinates alone — it can be frozen and run **zero-shot on a
montage never seen in pretraining**.

This is the *inductive* vs. *transductive* distinction (GraphSAGE): a codex fits
one embedding per electrode identity and cannot embed an electrode it never saw;
a shared function over structural inputs generalizes to unseen electrodes.

## 2. Architecture

- **Patcher → projection → spatial encoder → temporal Transformer**, standard
  EEGPT-style stack (~4.8 M online parameters).
- The only distinguishing component is the **spatial encoder**:
  - **Geometric:** multi-head attention with a shared MLP over `g_ij` added
    inside the attention computation. No per-electrode parameters.
  - **chind (baseline):** `geometry_injection='none'` — plain dot-product
    attention, no geometric term, no per-electrode parameters. Also
    montage-agnostic, so it can transfer zero-shot.
- **Descriptor.** Headline `g_ij = [p_i, p_j, p_i − p_j, ‖p_i − p_j‖] ∈ R¹⁰`
  (absolute positions + signed displacement + distance). The **pos_only**
  ablation keeps `[p_i, p_j] ∈ R⁶`. Coordinates are on a fixed scale from
  channel names (`src/geo/preprocess.ch_pos_from_names`), with no per-montage
  normalization — essential so the held-out montage's coordinates match the
  pretraining scale.
- **Primary configuration G3** (geometry enters at both the attention-score
  bias and the value path). An injection-site ablation (E5) crosses the site
  (G1 score-only / G2 value-only / G3 both) with the descriptor (full /
  pos_only); the value path carries the cross-montage signal (G1 is inert), and
  G3 is significant under both descriptors.
- **Objective.** Momentum-encoder alignment + masked-patch reconstruction
  (per-channel target, temporal-only masking, λ_R = 1, 50% mask).

## 3. Experiment

**Leave-one-montage-out, mixed-corpus pretraining; zero-shot linear probe on the
held-out montage.** Two scales:

- **Balanced n9** — 9 subjects each from the included datasets; all three
  held-out directions (Sleep-EDFx, BCIC-2B, PhysioNet MI). Directional only.
- **Scaled n156** — 78 PhysioNet MI + 78 Sleep-EDFx, held out BCIC-2B. The one
  split that is both balanced and large (any corpus *containing* BCIC-2B is
  capped at its 9 subjects). This is the powered headline test.

Datasets: PhysioNet MI (64-ch motor imagery), Sleep-EDFx (2 bipolar, sleep
staging), BCIC-2B (3-ch motor imagery). Preprocessing locked across datasets
(0.5–45 Hz, 200 Hz resample, 4 s epochs, per-epoch z-score). Metric: balanced
accuracy (LOSO or within-subject-night mean ± std). Significance: paired *t* +
Wilcoxon over shared folds.

## 4. Results

**Scaled n156, zero-shot BCIC-2B:**

| Variant | BAC | vs. chind |
|---|---|---|
| chind | 0.509 ± 0.016 | — |
| codex (name-aligned) | 0.522 ± 0.021 | point est. only |
| geometric (pos_only) | 0.540 ± 0.026 | +0.031, p=0.022, 7/9 |
| **geometric (full)** | **0.545 ± 0.027** | +0.036, p=0.023, 7/9 |

**Balanced n9** (directional, within noise): both geometric variants lead chind
on all 3/3 held-out montages; gaps ≤1.3 BAC points, not individually significant.

**Reading.** The geometric advantage emerges with pretraining scale (the
signature of an inductive bias) and is significant against the only comparable
zero-shot baseline. Both geometric variants also exceed the codex point estimate.

## 5. The codex baseline (name-aligned transfer)

The transductive codex is evaluated cross-montage by **name-aligned transfer**:
for each held-out electrode whose channel name exists in the pretraining montage,
copy that source codex row; otherwise random-init. For BCIC-2B all 3 channels
(C3/Cz/C4) matched the pretraining montage, so the codex received its full
trained embeddings — its best case — and still landed at 0.522, below both
geometric variants.

Caveat (conservative): the codex checkpoint was pretrained on PhysioNet only,
not the mixed corpus. That regime *favors* the codex (more in-domain data, fully
matched montage), so its underperformance against geometry is a comparison
biased toward the codex. A mixed-corpus codex pretrain is the clean follow-up.

## 6. Limitations

- One powered held-out montage (BCIC-2B); the other two directions are only
  testable at the underpowered n9 scale.
- Small eval cohort (BCIC-2B has 9 subjects); single seed for the headline.
- n156 is large for this project, small by foundation-model standards.
- Descriptor is moderate (no hemisphere/geodesic/anatomy terms).

## 7. Future work

1. A second distinct powered held-out montage (e.g. 128-ch MI, or 22-ch BCIC-2A).
2. Scale the corpus further and characterize the curve.
3. Mixed-corpus codex pretrain for an apples-to-apples codex bar.
4. Richer geometric descriptors under fixed G3.
5. Multi-seed the headline.
6. **Per-subject MRI-based cortical projection of electrode coordinates.** The current g_ij uses scalp-surface positions from a standard 10-20 template. A stronger variant would project p_i inward to the cortical surface using a subject-specific BEM head model (FreeSurfer segmentation + MNE pipeline): p_i' = p_i − d_i·n̂_i, where n̂_i is the outward scalp normal at electrode i and d_i is the ray-intersection depth to the cortical surface. This gives non-uniform, subject-specific depth corrections (frontal skull is thicker than temporal, varies across subjects), so the pairwise g_ij' better reflects true cortical source separation rather than scalp-surface proximity. A feasible intermediate step — requiring no new subject data — is a template-based version using the MNE fsaverage BEM, which already captures regional skull-thickness variation and is precomputable once for any standard montage. Full per-subject MRI requires a dataset pairing structural MRI with motor-imagery EEG; no such public dataset currently exists for the standard MI benchmarks.
