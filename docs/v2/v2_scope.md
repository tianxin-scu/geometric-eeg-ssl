# v2 Scope — Three Changes, One Headline

Branch: `v2-improvements`. v1 frozen at tag `v1-baseline`. New code under
`src/v2/`, new configs under `configs/v2/`. v1 modules are not edited.

**Headline experiment.** Zero-shot cross-montage transferability, measured
as 3-way leave-one-dataset-out: {phys+bcic→sleep}, {phys+sleep→bcic},
{bcic+sleep→phys}. Three BAC numbers. If v2 lifts zero-shot BAC over v1 and
v1 didn't, the thesis is told.

Everything in this doc serves that one experiment. No scope creep.

---

## Why v1 fell short on the headline

Full-mode v1 numbers (from `runs/eval/full/`):

| Eval | G1 | codex | chind |
|---|---|---|---|
| PhysioNet in-distribution | 0.342 | 0.341 | 0.333 |
| BCIC-2B zero-shot | **0.509** | **0.522** | 0.501 |
| Sleep-EDFx zero-shot | 0.532 | 0.533 | — |

In-distribution: G1 ≈ codex, as expected. Zero-shot: G1 *loses* to the
codex baseline with random fallback. The geometric story didn't land.

Diagnostic on the trained G1 `geom_bias_mlp` (sweep of `b(g)` across
distances, 7 canonical displacement directions):

- `b(d)` is smooth, monotonic, no saturation, no NaN at any distance in
  `[0, 2.2]`.
- Per-head pattern is the **same shape** at every distance — the function
  is essentially `b_h(d) ≈ α_h · d + β_h`.
- The descriptor's three displacement axes (`dx, dy, dz`) contribute
  almost nothing. v1's `b: ℝ⁴ → ℝ⁸` collapsed to a per-head linear ramp
  in distance.

So v1's architecture *can* express richer geometry but its training never
forced it to. The fix is not more inductive bias — it's removing the
bottlenecks that let the trivial solution win.

---

## The three changes

### v2.0 — Richer geometric descriptor + fixed-scale coordinates

**Sign-off required:** Tier-1 preprocessing change (`coord_normalize`) and
architectural change beyond proposal §3. Tianxin signed off in the v2
scoping conversation.

**What v1 does.** `g_ij = [||p_i − p_j||, p_i − p_j] ∈ ℝ⁴`. Translation-
invariant by construction — "two adjacent frontal electrodes" and "two
adjacent occipital electrodes" produce identical input to `b`. Coordinates
are normalized **per-montage** (centroid-subtract + divide by per-montage
max-norm), so the same physical Cz lands at different `p` values across
datasets. Cross-montage `g_ij` are not even on the same scale.

**What v2.0 does.**

```
g_ij = [p_i, p_j, p_i − p_j, ||p_i − p_j||] ∈ ℝ¹⁰
```

A strict superset of v1 — the MLP can recover v1 by zeroing the weights
on `p_i, p_j`. Lets `b` learn position-conditional structure (e.g. a
frontal pair behaves differently from an equidistant occipital pair) if
the training signal supports it.

Coordinates use **fixed-scale normalization**: subtract a reference
centroid and divide by a reference scale, both derived once from the
full standard_1005 layout (centroid ≈ (0.07, −1.68, 2.12) cm, scale ≈
12.76 cm). Any subset of electrodes projected into this frame yields
the same `p` for the same name across every dataset. This is a hard
prerequisite for `v2.0`: absolute coordinates are meaningless unless
they share a frame.

**Cost.** `geom_bias_mlp` input 4 → 10: +192 params/layer. Same for
`geom_value_mlp`. Total ≈ +800 params on a 4.8M-param backbone.
Negligible.

**Why this isn't hand-engineering.** v1's ℝ⁴ descriptor already imposes
translation invariance — that *is* hand-engineering. v2.0 removes that
imposition. The NN gets the raw coordinates and chooses what to use.
That's the point of having a learned `b`.

**Files.**
- `src/v2/model/geometric_attention_v2.py` — `GeometricSpatialEncoderV2`
  with the 10-D descriptor builder.
- `src/v2/preprocess.py` — `get_channel_positions_fixed` using the
  standard_1005-derived constants. Reused by all v2 dataset loaders.
- `configs/v2/pretrain/geometric_v2.yaml`.

### v2.1 — Per-channel alignment target

**What v1 does.** The backbone CLS-pools electrodes inside the spatial
encoder, so the momentum target `z_momentum ∈ ℝ^{B × T_p × d}` is
already a spatial average. `L_align = −2 · cos(pred, z_momentum)` regresses
to a per-time-step electrode-averaged summary. No gradient pressure on the
encoder to retain per-electrode structure. (`L_recon` target is also
explicitly `patches.mean(dim=1)` — same averaging.)

This is the precise reason v1's `b` collapsed to a function of `d`. With
the only target being a spatial pool, the encoder has no incentive to
route attention by direction — only by "how much to mix everything
together."

**What v2.1 does.** Replace the CLS pool with the per-electrode token
sequence. Align per-channel:

```
z_online   : (B, M, T_p, d)       # pre-pool, per-electrode
z_momentum : (B, M, T_p, d)       # same shape
L_align    = − 2 · mean cos(pred[b,m,t], z_momentum[b,m,t])
```

Drop spatial masking. Keep temporal masking (predict masked time steps
from unmasked ones). The per-channel target is what creates gradient
pressure on `b` to use `p_i, p_j, dx, dy, dz` — predicting electrode
*m*'s target requires routing attention by direction, not just distance.

**Cost.** No new parameters. Slightly different backbone forward (returns
per-electrode tokens instead of CLS pool). `L_recon` head needs to output
per-electrode patches, not channel-mean patches.

**Files.**
- `src/v2/model/backbone_v2.py` — Backbone returning per-electrode tokens.
- `src/v2/losses.py` — per-channel `L_align` + per-channel `L_recon`.
- `configs/v2/pretrain/geometric_v2.yaml` (same as v2.0; reused).

### v2.2 — Mixed cross-montage pretraining with subject + epoch capping

**What v1 does.** Pretrains on PhysioNet MI only. The geometric MLP only
ever sees `g_ij` from one montage.

**What v2.2 does.** Three leave-one-dataset-out pretraining runs:

| Pretrain set | Held-out eval target |
|---|---|
| PhysioNet + BCIC-2B | Sleep-EDFx |
| PhysioNet + Sleep-EDFx | BCIC-2B |
| BCIC-2B + Sleep-EDFx | PhysioNet |

**Two-level capping (Option A balancing, locked).** Subject-level cap
**N=9 per dataset** + epoch-level cap **4000 per dataset per pass**.

Why both levels: subject capping equalizes the *cross-subject diversity*
each dataset contributes; epoch capping equalizes the *gradient weight*
per pass. BCIC-2B has only 9 subjects total (hard ceiling), so 9 is what
PhysioNet and Sleep-EDFx are capped at to match — the smallest dataset
sets the bar for fairness.

Total pretrain corpus per run: 18 subjects. Small, but the per-channel
target + masked reconstruction objectives are sample-efficient enough.

**Three v2 variants, identical training conditions.** The fair-comparison
table is three head-to-head pretrain runs per leave-one-out split:

- **`v2_geometric`** — `GeometricSpatialEncoderV2(g3, g_dim=10)`. The headline.
- **`v2_chind`** — same encoder with `geometry_injection='none'`. The "no spatial structure" baseline.
- **`v2_codex`** — `TransductiveSpatialEncoderV2` (per-montage codex). Deferred; see `docs/v2/experiment_protocol.md` §codex.

Only the spatial encoder differs across variants; everything else
(architecture, optimizer, schedule, mask scheme, EMA, loss, subject cap,
epoch cap) is identical. The eval is zero-shot on the held-out dataset.

**v1 baselines are NOT in the headline comparison.** v1 was pretrained
on 105 PhysioNet subjects only — a different regime. Putting v1's
0.509/0.532 zero-shot numbers next to v2's 9+9+9 numbers would
mis-attribute any difference to the v2 architecture when it could be
due to the pretraining corpus change. If we want a v1-style comparison,
it requires retraining v1 at matched 9-subject scale (a separate
experiment, only worth doing if v2 doesn't already win cleanly).

**Files.**
- `src/v2/datasets/mixed_corpus.py` — round-robin sampler with per-dataset
  epoch cap. Subject capping happens upstream in `scripts/v2_pretrain.py`
  via `--n-subjects-per-dataset`.
- `scripts/v2_pretrain.py` — `--variant {geometric,chind}` selects the
  spatial encoder; `--pretrain-datasets X,Y` picks the leave-one-out pair.
- `docs/v2/experiment_protocol.md` — canonical run inventory + eval recipe.

**Why capped equal.** The claim is *coverage of `g_ij` matters; count
doesn't*. With unbounded mixing, Sleep-EDFx (≈100× more epochs than the
others) would dominate and the run would mostly look like single-montage
pretraining on Sleep. Capping equalizes per-dataset gradient contributions
and keeps wall-clock bounded.

**Cost.** ~2–3× the per-run wall-clock of a single-dataset v1 run
(two datasets × N epochs each), times 3 runs. ~6–9× v1 pretraining cost.
Manageable on Colab.

**Files.**
- `src/v2/datasets/mixed_corpus.py` — round-robin sampler with
  per-dataset epoch cap.
- `configs/v2/pretrain/mixed_no_sleep.yaml`,
  `mixed_no_bcic.yaml`, `mixed_no_phys.yaml`.

---

## Suggested order

The three changes reinforce each other; partial v2 may not lift numbers.

1. **v2.0 first** (descriptor + normalizer). Strict superset of v1; if it
   ever hurts in-distribution BAC vs v1, we know the implementation is
   wrong.
2. **v2.1 second** (per-channel target). Provides the gradient signal that
   makes v2.0 useful. Test against v2.0-only as ablation.
3. **v2.2 last** (cross-montage mixing). The actual headline. With v2.0+v2.1
   in place, this is where the cross-montage `g_ij` exposure shows up.

If pressed for time: **v2.0 + v2.2 without v2.1** may still show a
modest lift (richer descriptor + multi-montage exposure). v2.1 alone
without v2.0 has nothing for `b` to learn — the descriptor would still
collapse to `d`. v2.2 alone without v2.0 hits the same per-montage
normalization wall.

## Headline numbers to report

Three rows in a table:

| Pretrain | Eval (held-out) | v1 BAC | v2 BAC | Δ |
|---|---|---|---|---|
| phys+bcic | Sleep-EDFx | (re-eval v1) | (v2 run) | |
| phys+sleep | BCIC-2B | (re-eval v1) | (v2 run) | |
| bcic+sleep | PhysioNet | (re-eval v1) | (v2 run) | |

v1 baseline for each row uses single-dataset pretraining on the
larger of the two corresponding sources (so v1 baseline is "best v1
could do without mixing"). If v2 lifts ≥2 of 3 rows, the thesis lands.

## Explicitly out of scope (still)

- Spherical / geodesic descriptors — `project_direction_summary.md` §4 O6.
- Attention sparsification — §4 O7.
- GNN reformulation — `CLAUDE.md` constraint 2.
- New datasets beyond the three already in use.
- Synthetic per-subject coordinate augmentation (closed-loop; proves
  nothing about real transfer).
- Multi-head geometric bias — already in v1, false alarm in the prior
  draft of this doc.
- Masked-patch reconstruction — already in v1, false alarm in the prior
  draft of this doc.

## What replaces the load-bearing constraint list

`CLAUDE.md` still applies. The v2.0 change adds one new constraint
worth recording once we ship it: **coordinates use fixed-scale
normalization in any code path the geometric MLP touches.** Mixing
per-montage and fixed-scale coordinates within a run silently corrupts
`p_i, p_j`. Worth promoting to `copilot_standing_instructions.md`
after v2.0 lands.
