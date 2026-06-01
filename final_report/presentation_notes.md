# Presentation Notes — Geometrically Inductive SSL for EEG

## Slide 2 — Problem statement

### What is a montage, and why is it on this slide?

A **montage** is the specific electrode placement configuration for a recording session: how many electrodes, where they sit on the scalp (expressed as 3D Cartesian coordinates), and what they're called. Different research groups and BCI devices use completely different montages:

| Dataset | Channels | Configuration |
|---|---|---|
| PhysioNet MI | 64 | 10-10 system, dense coverage |
| BCIC-2B | 3 | C3, Cz, C4 — motor-cortex strip only |
| Sleep-EDFx | 2 | Fpz-Cz and Pz-Oz — clinical frontal/occipital |

### How the three montages relate geometrically (answered)

v2 uses a **shared coordinate frame**: every channel's position is looked up from the same `standard_1005` template and normalized by one fixed centroid/scale ([src/v2/preprocess.py](src/v2/preprocess.py)). So an electrode **name maps to the same (x,y,z) in every dataset** — a given name does *not* move between datasets (that's v1's per-montage behavior, fixed in v2).

- **BCIC-2B ⊂ PhysioNet, exactly.** C3/Cz/C4 are byte-identical across the two and all live in PhysioNet's 64:
  - C3 = (−0.518, 0.041, 0.338), Cz = (−0.002, 0.060, 0.619), C4 = (0.521, 0.046, 0.332).
  - BCIC-2B is literally a **3-electrode geometric subset** of PhysioNet.
- **Sleep-EDFx is the genuinely different montage.** Its channels are *bipolar* derivations placed at the **midpoint** of their reference pair, so they don't coincide with any single electrode in the others:
  - `FpzCz` = midpoint(Fpz, Cz) = (−0.003, 0.442, 0.220); nearest PhysioNet electrode = Fz at dist 0.201.
  - `PzOz` = midpoint(Pz, Oz) = (−0.004, −0.636, 0.215); nearest = Poz at dist 0.036.
  - Shares the *frame*, occupies distinct *points*.

**Crucial nuance for Q&A.** Headline run = pretrain on PhysioNet + Sleep, test on BCIC-2B. Because C3/Cz/C4 (and their pairwise `g_ij`) already appear in PhysioNet, **the positions BCIC presents at test were already seen in pretraining.** What is novel at test is the *montage configuration* (only 3 electrodes, attention over just those 3), **not the coordinates**. The genuinely-novel-position case is when **Sleep-EDFx** is held out (its bipolar midpoints are unseen) — that is one of the small-corpus directional runs, not the headline. Expect a reviewer to probe this; the honest framing is "unseen electrode *set/count*, overlapping positions."

### Is "montage" really non-stationarity?

Strictly, no — not in the time-series statistics sense (where non-stationarity means the mean or variance shifts over time). Montage mismatch is really a **structural distribution shift**: the input dimensionality $M$ changes, electrode identities change, and the spatial sampling of the scalp changes.

It appears on the same slide as session and subject drift because all three break the same assumption that underpins codex and channel-independent models: **that electrode $X$ means the same thing across recordings.** Once you break that assumption — whether because the cap shifted (session), the head is smaller (subject), or there are only 3 electrodes instead of 64 (montage) — any model parameter indexed by electrode identity becomes invalid.

The geometry-conditioned approach resolves all three with a single mechanism: replace the electrode identity index with a 3D scalp coordinate as input. Session cap shifts become small perturbations in $p_i$; subject head-size differences are captured by the absolute scale of $p_i$; montage differences are just different sets of $p_i$, all handled by the same shared MLP.

### How to say it in the talk (30 seconds)

> "I'm calling these 'geometric non-stationarities' — not because the statistics of the signal are shifting, but because the physical geometry of the recording is changing. Session drift is small: the cap moves a few millimeters. Subject variation is medium: heads are different sizes so C3 doesn't land on the same cortical tissue. Montage change is categorical: you go from 64 channels to 3. All three break any model that memorizes electrode identities. Geometry as input fixes all three."

---

## Slide 3 — Solution

- $g_{ij} \in \mathbb{R}^{10}$: the two 3D positions $p_i, p_j$ (6 dims) + displacement $p_i - p_j$ (3 dims) + scalar distance $\|p_i - p_j\|$ (1 dim).
- G3 = geometry injected at both the attention score and the value projection — ablated in prior work, strongest variant.
- pos_only = $[p_i, p_j] \in \mathbb{R}^6$ only, no displacement or distance — less information, still significant.

### The two G3 equations (in case anyone asks)

Per head $h$, geometry enters at two points via **two separate MLPs**:

**1. Score-side bias** (`geom_bias_mlp`, $\mathbb{R}^{10}\to\mathbb{R}^{H}$):
$$e_{ij}^{h} = \frac{\langle q_i^{h},\, k_j^{h}\rangle}{\sqrt{d_h}} + b^{h}(g_{ij}), \qquad a_{ij}^{h} = \mathrm{softmax}_j\bigl(e_{ij}^{h}\bigr)$$

**2. Value-side increment** (`geom_value_mlp`, $\mathbb{R}^{10}\to\mathbb{R}^{d}$):
$$o_i^{h} = \sum_j a_{ij}^{h}\,\bigl(v_j^{h} + w^{h}(g_{ij})\bigr)$$

- $b(g_{ij})$ and $w(g_{ij})$ are outputs of **different** MLPs, both fed the same $g_{ij}$.
- $v_j$ comes from the $q,k,v$ projection of the patch — **not** from a MLP; the value MLP only adds the geometric term $w(g_{ij})$.
- Neither geometric term ever touches the patch embedding directly; both enter *inside* attention. G1 = bias only, G2 = value only, **G3 = both** (the only mode v2 ran).
- Code: [src/v2/model/geometric_attention_v2.py](src/v2/model/geometric_attention_v2.py) lines 230 (bias) and 254–255 (value).

---

## Slide 4 — Headline result

**Protocol to explain verbally** (the dedicated experiment-design slide was cut for time):

- **Leave-one-montage-out:** pretrain on a mix of two montages, hold out the third dataset entirely, freeze the encoder, fit a linear probe on the held-out montage. "Zero-shot" = the encoder never saw that montage's electrode layout during pretraining.
- **Fair comparison:** identical training for every variant — only the spatial encoder differs (geom / pos_only / chind). Same data, same recipe.
- The codex is **completely absent** — it cannot be run at all on an unseen montage, not merely that it performs worse.

**The result** (the slide shows only the three bars + the pretrain→test line; everything below is for verbal delivery / Q&A):

- $n=156$ pretrain: 78 PhysioNet + 78 Sleep-EDFx subjects. BCIC-2B is the held-out montage ($n=9$ LOSO).
- The three bars: **chind = 0.509**, **pos_only = 0.531**, **geom (full) = 0.545**.
- Geometric (full) **+0.036** over chind, $p=0.023$ (paired *t*), Wilcoxon $p=0.027$, **7/9** subjects, $d_z=0.93$, 95% CI $[+0.006, +0.066]$.
- pos_only also significant (**+0.022**, $p=0.013$, **8/9**, $d_z=1.07$) — the result is not fragile to the descriptor choice.
- **Chance = 0.50** (binary; BCIC-2B is left- vs right-hand motor imagery). The y-axis is zoomed to $[0.49, 0.56]$, so read the bars as differences above chance, not absolute heights.
- **Codex is absent by necessity** — it cannot run on an unseen montage at all (no embedding slot for BCIC-2B's electrodes), so there is no codex bar to show. That absence is the point.

### What is BAC, and how is it computed?

- **BAC = balanced accuracy** = the macro-average of per-class recall: $\mathrm{BAC} = \frac{1}{K}\sum_{k=1}^{K}\mathrm{recall}_k$, where $\mathrm{recall}_k = \frac{\text{correctly predicted class-}k}{\text{true class-}k}$.
- For the binary BCIC-2B task this is just $\tfrac{1}{2}(\text{sensitivity} + \text{specificity})$.
- Computed with `sklearn.metrics.balanced_accuracy_score(y_true, y_pred)` ([scripts/v2_probe.py:235](scripts/v2_probe.py#L235)).
- **Why balanced (not plain) accuracy:** it corrects for class imbalance — a model that just predicts the majority class scores at chance ($1/K$), not at the majority-class frequency. So **chance = 0.50** for binary regardless of class skew.
- **Per-subject → headline:** BAC is computed per held-out subject (LOSO fold), then averaged over the 9 BCIC-2B subjects. The significance tests are paired over those 9 per-subject BAC values (geom vs chind, same folds).

### TODO (tonight) — add a REAL confusion matrix

The slide currently shows only the three bars. A real confusion matrix is **not** in any saved artifact: the probe computes `y_pred` but discards it, saving only BAC / κ / weighted-F1 scalars ([scripts/v2_probe.py:233-249](scripts/v2_probe.py#L233-L249)). The n=156 checkpoint lives on another machine (not this one, not Colab).

To produce it tonight on the machine with the checkpoint:

1. Patch `_compute_metrics` / `loso_eval` in [scripts/v2_probe.py](scripts/v2_probe.py) to also save `sklearn.metrics.confusion_matrix(y_true, y_pred)` per subject (and ideally raw `y_true`/`y_pred`).
2. Re-run the **n=156 → BCIC-2B** probe for `v2_geometric`.
3. Aggregate the confusion matrix by **summing** the per-subject 2×2 matrices (pooled), or report the per-subject mean — decide which and label it.
4. Then we drop the real matrix onto slide 4 next to the bars.

### Illustrative confusion matrix (teaching aid only — NOT the real result)

Use this only to *explain BAC* if asked; do not present it as the experiment outcome. Binary left/right MI, 10 trials per class:

|                    | Pred Left | Pred Right |
|--------------------|-----------|------------|
| **True Left (10)** | 8         | 2          |
| **True Right (10)**| 4         | 6          |

Recall(Left) = 8/10 = 0.8, Recall(Right) = 6/10 = 0.6 → BAC = (0.8+0.6)/2 = **0.70**. (Equivalently: sensitivity 0.8, specificity 0.6.) The real headline numbers are chind 0.509 / pos_only 0.531 / geom 0.545 — much closer to chance (0.50) than this toy.

> **Note:** the small balanced 9-subject-per-dataset run (directional, geom $\geq$ chind on 3/3 held-out montages but within noise) was on the cut slide. It still exists in `results/v2/` and `report.md` if a question demands it — but the headline rests on the powered $n=156$ run, not on that.

---

## Likely Q&A

**Q: Why not just fine-tune the codex on the new montage?**
Fine-tuning requires labeled data on the target montage, which defeats the zero-shot transfer goal. The whole point is that a clinician can plug in a new headset with different channels and the model runs immediately.

**Q: Is 3D position always available?**
Standard montage files (MNE's built-in 10-10, 10-20 layouts) provide 3D coordinates. Custom devices need digitization, but that's one measurement per deployment, not per subject.

**Q: Why not train separate models per montage?**
That is exactly the codex/chind approach, implicitly. It requires either enough data per montage to train from scratch, or a separate fine-tuning step. The geometric model shares weights across all montages.

**Q: The gaps look small — does this actually matter?**
At $n=9$ subjects the underpowered runs look nearly flat. The $n=156$ run shows the gap grows with data and becomes significant. The trend is the finding, not a single number.
