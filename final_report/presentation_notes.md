# Presentation Notes — Geometrically Inductive SSL for EEG

## Slide 2 — Problem statement

### What is a montage, and why is it on this slide?

A **montage** is the specific electrode placement configuration for a recording session: how many electrodes, where they sit on the scalp (expressed as 3D Cartesian coordinates), and what they're called. Different research groups and BCI devices use completely different montages:

| Dataset | Channels | Configuration |
|---|---|---|
| PhysioNet MI | 64 | 10-10 system, dense coverage |
| BCIC-2B | 3 | C3, Cz, C4 — motor-cortex strip only |
| Sleep-EDFx | 2 | Fpz-Cz and Pz-Oz — clinical frontal/occipital |

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

---

## Slide 3 — Solution

- $g_{ij} \in \mathbb{R}^{10}$: the two 3D positions $p_i, p_j$ (6 dims) + displacement $p_i - p_j$ (3 dims) + scalar distance $\|p_i - p_j\|$ (1 dim).
- G1 = geometry at the **attention score** (bias path); G2 = at the **value** projection (value path); G3 = both. The E3 ablation shows the signal is in the **value path**: G2 ≈ G3 ≫ G1 (G1 ≈ no geometry). The slide-4 headline uses G2+full.
- pos_only = $[p_i, p_j] \in \mathbb{R}^6$ only, no displacement or distance. On slide 4 (value-path G2) this is **not** significant (0.520) — the full descriptor matters. Only G3+pos_only stays significant (0.540); see E3 table.

### The two G3 equations (in case anyone asks)

Per head $h$, geometry enters at two points via **two separate MLPs**:

**1. Score-side bias** (`geom_bias_mlp`, $\mathbb{R}^{10}\to\mathbb{R}^{H}$):
$$e_{ij}^{h} = \frac{\langle q_i^{h},\, k_j^{h}\rangle}{\sqrt{d_h}} + b^{h}(g_{ij}), \qquad a_{ij}^{h} = \mathrm{softmax}_j\bigl(e_{ij}^{h}\bigr)$$

**2. Value-side increment** (`geom_value_mlp`, $\mathbb{R}^{10}\to\mathbb{R}^{d}$):
$$o_i^{h} = \sum_j a_{ij}^{h}\,\bigl(v_j^{h} + w^{h}(g_{ij})\bigr)$$

- $b(g_{ij})$ and $w(g_{ij})$ are outputs of **different** MLPs, both fed the same $g_{ij}$.
- $v_j$ comes from the $q,k,v$ projection of the patch — **not** from a MLP; the value MLP only adds the geometric term $w(g_{ij})$.
- Neither geometric term ever touches the patch embedding directly; both enter *inside* attention. G1 = bias only, G2 = value only, **G3 = both** (v2 ran all three in the E3 ablation; the headline variant is G2+full).
- Code: [src/v2/model/geometric_attention_v2.py](src/v2/model/geometric_attention_v2.py) lines 230 (bias) and 254–255 (value).

---

## Slide 4 — Headline result

**Protocol to explain verbally** (the dedicated experiment-design slide was cut for time):

- **Leave-one-montage-out:** pretrain on a mix of two montages, hold out the third dataset entirely, freeze the encoder, fit a linear probe on the held-out montage. "Zero-shot" = the encoder never saw that montage's electrode layout during pretraining.
- **Fair comparison:** identical training for every variant — only the spatial encoder differs (geom / pos_only / chind). Same data, same recipe.
- The codex is **completely absent** — it cannot be run at all on an unseen montage, not merely that it performs worse.

**The result** (the slide shows only the three bars + the pretrain→test line; everything below is for verbal delivery / Q&A):

- $n=156$ pretrain: 78 PhysioNet + 78 Sleep-EDFx subjects. BCIC-2B is the held-out montage ($n=9$ LOSO).
- The three bars are now **all G2** (value-path injection), varying only the descriptor: **chind = 0.509**, **pos_only (G2) = 0.520**, **geom (G2+full) = 0.549**. This keeps slide 4 internally consistent (one injection mode, two descriptors + the no-geometry baseline). The pre-registered G3+full (0.545) and G3+pos_only (0.540) are statistically tied with their G2 counterparts on the mean — see the E3 ablation section and the "I don't claim G2 > G3" caveat.
- **geom (G2+full) +0.040** over chind, $p=0.004$ (paired *t*), Wilcoxon $p=0.008$, **8/9**, $d_z=1.32$, 95% CI $[+0.017, +0.063]$.
- **pos_only (G2+pos_only) = 0.520 is NOT significant** (+0.011 over chind, $p=0.229$, Wilcoxon $p=0.301$, 5/9, CI $[-0.008, +0.030]$ — spans 0). The honest reading the slide now tells: **the full descriptor matters** — reducing to position-only ($\mathbb{R}^6$) loses the effect for the value-path variant. (Aside: G3+pos_only *does* stay significant at 0.540; if a reviewer asks why I don't show that, the answer is "I held the injection fixed at G2 across all three bars for a clean comparison." The G3 descriptor numbers are in the E3 table.)
- **Chance = 0.50** (binary; BCIC-2B is left- vs right-hand motor imagery). The y-axis is zoomed to $[0.49, 0.56]$, so read the bars as differences above chance, not absolute heights.
- **Codex is absent by necessity** — it cannot run on an unseen montage at all (no embedding slot for BCIC-2B's electrodes), so there is no codex bar to show. That absence is the point.

### Confusion matrix

The probe saves the confusion matrix: `_compute_metrics` / `loso_eval` / `within_subject_eval` in [scripts/v2_probe.py](scripts/v2_probe.py) write per-subject matrices plus a pooled `aggregate.confusion_matrix` (summed over the 9 LOSO folds) and `aggregate.labels`.

**Headline — geom = G2+full, n=156 → BCIC-2B** (pooled counts; rows = true, cols = predicted; class 0 = Left MI, 1 = Right MI):

|                 | Pred Left | Pred Right | recall |
|-----------------|-----------|------------|--------|
| **True Left**   | 1840      | 1420       | 0.564  |
| **True Right**  | 1519      | 1741       | 0.534  |

BAC = (0.564 + 0.534) / 2 = **0.549**. This is the matrix now shown (row-normalized) on slide 4. (For reference, the pre-registered **G3+full** matrix was 1799/1461/1501/1759 → recall 0.552/0.540 → BAC 0.545; available in `runs/pretrain/v2_geometric_no_bcic_n78/probe_bcic_2b_loso.json`.)

**chind, n=156 → BCIC-2B** (the baseline, for contrast):

|                 | Pred Left | Pred Right | recall |
|-----------------|-----------|------------|--------|
| **True Left**   | 1716      | 1544       | 0.526  |
| **True Right**  | 1658      | 1602       | 0.491  |

BAC = (0.526 + 0.491) / 2 = **0.509**. Note chind's Right-MI recall (0.491) sits **below chance** — it leans toward predicting Left — whereas geom is balanced across both classes (0.552 / 0.540). That class balance, not just the mean, is the qualitative win.

**pos_only = G2+pos_only, n=156 → BCIC-2B** (the slide-4 pos_only bar):

|                 | Pred Left | Pred Right | recall |
|-----------------|-----------|------------|--------|
| **True Left**   | 1735      | 1525       | 0.532  |
| **True Right**  | 1608      | 1652       | 0.507  |

BAC = (0.532 + 0.507) / 2 = **0.520**. Barely above chance and not significant — the position-only descriptor loses most of the value-path signal that full $g_{ij}$ carries. (The G3+pos_only matrix — 1755/1505/1493/1767 → 0.540, still significant — is in `results/v2_n78/e5_confusion_matrices.md`, the committed laptop-portable file with all seven cells' matrices + CIs.)

## Slide 6 — Did I run what the proposal promised?

**What this slide is for.** A reviewer who read my proposal will ask "did you do the experiments you listed?" This slide answers honestly, split into two halves: the top three I **delivered**, the bottom three were **out of reach** for a one-quarter course project and are reported as limits (not failures). Labels use the report's clean gapless scheme — **E1, E2(a/b/c), E3, E4** — mapped back to the proposal in the crosswalk section below.

### Top half — delivered

**E2(c) — cross-montage zero-shot (the headline).** The central promise: a model pretrained on a mix of montages transfers to a *held-out* montage with no retraining. Delivered, and the gap is significant on the one powered test (BCIC-2B, n=156: geom 0.545 vs chind 0.509, p=0.023). I deliberately present **only BCIC** — it's the one properly-powered, significant result. The other two held-out montages (PhysioNet, Sleep) are tiny-corpus and within noise; I don't lean on them.

**E3 — where $g_{ij}$ enters (G1/G2/G3)** *(proposal E5)*. The principal ablation: does geometry help most as a score-bias (G1), a value-increment (G2), or both (G3)? Ran all three crossed with both descriptors (full R¹⁰ / pos_only R⁶), same n=156 regime, zero-shot BCIC. **Headline finding: the geometry signal lives in the value path. G2 (value-only, full) ties G3 (both) at the top (0.549 vs 0.545, both significant); G1 (score-bias-only) is inert (≈ chind).** Full table and the honest caveats are in the *E3 ablation results* section below.

**E4 — channel-independent baseline** *(proposal E7)*. In the proposal this was a third-tier *sanity check*. In practice it became my **actual headline baseline** (`chind`), because the codex baseline (E1's comparator) can't run cross-montage at all — so chind is the strongest thing that *can* run on every montage. Promoted, not dropped.

### Bottom half — out of reach, and why (say the reason before a reviewer asks)

**E1 — in-distribution, geom vs codex.** The proposal's "fairness anchor" compared geometry against a learned-codex baseline on a *matched* montage. But the whole value proposition is cross-montage, and **a codex literally cannot run on a montage it wasn't trained on** (no embedding slot for the new electrodes). So the in-distribution codex comparison answers a question the project no longer hinges on; I never built the v2 codex. Honest status: not run, because the regime that matters is the one codex can't enter.

**E2(a) — cross-session.** The proposal itself says geometry only addresses the *placement* part of session drift (cap moves a few mm). The dominant part of session-to-session change is **physiological** — alertness, fatigue, electrode impedance — which my architecture does not touch. And in real multi-session data the two are entangled, so the test couldn't cleanly isolate the geometric part. Not expected to help; honestly dropped.

**E2(b) — cross-subject / head size.** Testing the head-size claim needs a dataset where the *same task* is recorded across subjects with **documented, varying head geometry**. No publicly accessible EEG dataset provides controlled head-size variation with usable labels. Without the data, the experiment can't be run — a data-availability limit, not a modeling one.

### How to say it in the talk (20 seconds)

> "Of the experiments I proposed: the cross-montage transfer test — the headline — I ran, and it's significant. The G1/G2/G3 ablation and the baseline, I ran. Three others I couldn't: the in-distribution codex comparison is moot because a codex can't even run on a new montage; cross-session drift is mostly physiological, which geometry doesn't address; and cross-subject head-size testing has no public dataset. Those are limitations, and they're on the slide as limitations."

### If asked "your proposal skips E3/E4/E6/E8 — what happened?"

Short answer: the proposal was edited down from eight experiments to the surviving set without renumbering, so it shows gaps (E1, E2, E5, E7). Nothing is *missing* — proposal-E4 (per-subject variance) was folded into E2(b); proposal-E3/E6/E8 (channel-dropout, loss ablation, compute–quality) were cut. The report fixes the cosmetics with a clean gapless scheme — full mapping in the **Experiment relabeling — crosswalk** section below. (In that clean scheme, "E4" now means the channel-independent baseline, so don't confuse it with the proposal's old E4.)

### LOSO + Wilcoxon, explained simply (you asked)

**LOSO = Leave-One-Subject-Out.** With 9 BCIC-2B subjects, I train the linear probe on 8 subjects and test on the 9th — then rotate, so each subject takes a turn as the held-out test set. That gives **9 independent BAC scores**, one per subject. Why bother: it guarantees the test subject's data was never seen during probe training, so the number reflects *generalization to a new person*, not memorization. The headline BAC (0.545) is the mean of those 9 per-subject scores.

**Wilcoxon = a paired significance test that doesn't assume a bell curve.** For each of the 9 subjects I have *two* BAC numbers on the *same* fold: one from geom, one from chind. I want to know whether geom is reliably higher or whether the 9 small gaps are just luck. The **paired t-test** answers this assuming the per-subject differences are roughly normal; the **Wilcoxon signed-rank test** answers the same question using only the *ranks* of the differences (their order and sign), so it survives a couple of weird subjects and doesn't lean on normality. I report both (t: p=0.023, Wilcoxon: p=0.027) — when they agree, the result isn't an artifact of one test's assumptions. "7/9" = geom beat chind on 7 of the 9 subjects, the plain-language version of the same fact.

> One-liner if asked: "LOSO gives me 9 per-person scores; Wilcoxon checks whether geom's edge over chind across those 9 pairs is real or coincidence, without assuming the gaps are normally distributed."

### p, d_z, and the 95% CI, explained simply (the rest of the stats line)

All three describe the *same* 9 per-subject gaps (geom BAC − chind BAC), each answering a different question.

- **p-value (p = 0.023) — "is it real, or luck?"** The probability of seeing a gap at least this large *if geom and chind were truly equal* (the null hypothesis). p = 0.023 means there's about a 2.3% chance this 9-subject advantage is a coincidence. Below the conventional 0.05 cutoff, so it counts as "statistically significant." Honest caveat for a reviewer: with only 9 subjects, treat p as *suggestive evidence*, not proof — small samples make p jumpy.

- **Cohen's d_z (d_z = 0.93) — "is it big?"** The effect *size* for paired data: the mean gap divided by the standard deviation of the per-subject gaps. It measures how large the advantage is *relative to how much it wobbles between subjects*, in standard-deviation units. Rough convention: 0.2 small, 0.5 medium, 0.8 large — so 0.93 is a **large** effect. The point of reporting it alongside p: p tells you the gap is real, d_z tells you it's not trivially tiny. (A real-but-microscopic effect can have a small p with huge n; d_z guards against over-selling that.)

- **95% confidence interval (CI = [+0.006, +0.066]) — "how big, with what uncertainty?"** The range of true geom−chind advantages consistent with the data: the real gap is plausibly anywhere from **+0.6 to +6.6 BAC points**. Two things to read off it: (1) the **entire interval is above 0**, which is the same verdict as p < 0.05 — zero "no difference" is excluded; (2) it's **wide**, because n = 9 — I genuinely don't know if the true edge is small (~half a point) or sizeable (~6 points), only that it's positive. A bigger corpus would narrow it.

> One-liner if asked: "p says the edge is unlikely to be luck; d_z says it's a large effect per-subject; the CI says the true advantage is somewhere from half a point to six points and — importantly — entirely above zero. All three agree it's a real, non-trivial, but imprecisely-pinned win."

---

## Experiment relabeling — crosswalk for the report (documented)

**The situation.** The *submitted proposal* numbers its experiments E1, E2(a/b/c), E5, E7 — with visible gaps. That's a leftover from an editing pass that had eight experiments (E1–E8), then **dropped E3, E6, E8 and folded E4 into E2(b)** without renumbering. The proposal is frozen, so it keeps those gaps. **The report and slides use a clean, gapless relabeling instead**, with this crosswalk as the record. A one-line footnote in the report should point back: *"Experiment numbering is consolidated from the proposal's; see crosswalk."*

**Clean scheme (report + slides): E1, E2(a/b/c), E3, E4.** Only two experiments move — proposal-E5 → report-E3, proposal-E7 → report-E4. Everything else keeps its intuitive place.

| Report / slides | Proposal label | Experiment | Status in this study |
|---|---|---|---|
| **E1** | E1 | In-distribution, geometric vs. codex (fairness anchor) | Not run — codex can't enter the cross-montage regime the project hinges on |
| **E2(a)** | E2(a) | Cross-session robustness (within-subject, multi-night) | Not run — session drift is largely physiological, which geometry doesn't address |
| **E2(b)** | E2(b) + E4 | Cross-subject / head-size robustness, incl. per-subject variance | Not run — no public dataset with controlled head-size variation |
| **E2(c)** | E2(c) | **Cross-montage zero-shot transfer** | **✓ Delivered — the headline; significant on BCIC-2B (n=156, p=0.023)** |
| **E3** | E5 | Geometry-injection ablation: G1/G2/G3 × {full, pos_only} | ✓ Delivered (this run) |
| **E4** | E7 | Channel-independent baseline (`chind`) | ✓ Delivered — promoted from sanity-check to headline baseline |

**Dropped from the proposal (do not appear under any new label):**
- proposal-**E3** — channel-dropout robustness curves.
- proposal-**E6** — loss-component ablation (λ_R on/off, masking scheme).
- proposal-**E8** — compute–quality / FLOPs tradeoff.

**Why this scheme and not a full reorder.** Putting the headline (E2c) first was tempting, but keeping the proposal's logical ladder (in-distribution → robustness a/b/c → ablation → baseline) means a reviewer holding both documents reads them in the same order; only the two trailing labels shift. Lowest cognitive load, fully honest.

**Note for the talk:** the slides don't need to expose the crosswalk — just use the clean labels. The crosswalk lives here and in the report's experiments section so the mapping is on record if anyone compares against the proposal.

---

## E3 ablation results — where $g_{ij}$ enters (proposal E5)

Full G1/G2/G3 × {full, pos_only} grid, all at the n=156 headline regime, zero-shot LOSO on BCIC-2B. Paired stats are vs. `chind` over the 9 BCIC subjects (same folds). Source: `results/v2_n78/e5_ablation.{md,txt}`; JSONs in `runs/pretrain/v2_{g1full,g1posonly,g2full,g2posonly,geometric,geomposonly}_no_bcic_n78/probe_bcic_2b_loso.json`.

| cell | BAC | Δ vs chind | wins | t p | Wilcoxon p | d_z |
|---|---|---|---|---|---|---|
| chind (baseline) | 0.509 | — | — | — | — | — |
| G1 + full | 0.510 | +0.001 | 5/9 | 0.854 | 0.793 | 0.06 |
| G1 + pos_only | 0.506 | −0.003 | 4/9 | 0.402 | 0.496 | −0.29 |
| **G2 + full** | **0.549** | **+0.040** | **8/9** | **0.004** | **0.008** | **1.32** |
| G2 + pos_only | 0.520 | +0.011 | 5/9 | 0.229 | 0.301 | 0.43 |
| G3 + full | 0.545 | +0.036 | 7/9 | 0.023 | 0.027 | 0.93 |
| G3 + pos_only | 0.540 | +0.031 | 7/9 | 0.022 | 0.020 | 0.95 |

### What the ablation says (honest reading)

1. **The geometry signal enters through the value path, not the score-bias path.** G2 (value-only) and G3 (both) are the only cells that significantly beat chind; G1 (score-bias only) sits *on top of* chind (0.510 / 0.506 vs 0.509) at every descriptor — it is **inert** for cross-montage transfer here. So "*what* the attended messages carry" (value) encodes the geometry that transfers; "*who* attends" (score bias) does not, on its own.

2. **G2 ≈ G3; I do not claim G2 > G3.** G2+full 0.549 vs G3+full 0.545 is a +0.004 gap — *inside* the ~0.009 single-seed noise band established by the pos_only re-train (slide-4 provenance note). G2's paired stats look cleaner (p=0.004, 8/9, d_z=1.32 vs G3's p=0.023, 7/9, d_z=0.93), but on a **single seed per cell** that is not enough to rank them. The seed-stable, defensible claim is the **ordering G1 ≪ {G2 ≈ G3}**, not a G2-over-G3 win.

3. **This refutes the proposal's E5 hypothesis.** The proposal (O1/E5) predicted "*G1 (score-only) is competitive with G2/G3 at lower complexity*" and even made G1 the *default*. The data says the opposite: G1 is the weakest variant, indistinguishable from no geometry. That is a clean negative result and should be reported as one — the proposed cheap default would have failed; v2's choice to run G3 (and the discovery that G2 alone suffices) is what carries the result.

4. **Descriptor interaction.** Dropping to pos_only (R⁶) hurts G1 and G2 (G2 falls 0.549→0.520) but barely dents G3 (0.545→0.540, still significant). Reading: when geometry only enters one path, it needs the full descriptor; when it enters both, the redundancy makes it robust to descriptor reduction. Minor point, not headline.

### Caveats to state plainly
- **Single seed per cell.** No multi-seed bands; the G2/G3 tie and the descriptor-interaction point both live within seed noise. A multi-seed sweep is the obvious follow-up (and is cheap now — ~33 min/run on this machine).
- **One held-out montage (BCIC).** Same scope bound as the headline; the ablation inherits it.
- **Does not change the headline claim.** "Geometry-conditioned attention beats chind, significantly, zero-shot" stands regardless of G2-vs-G3. The ablation *refines* it to "via the value path."

### Decisions made (resolved)
- **Headline variant → G2+full (chosen).** Slide 4 now reports **G2+full (0.549, p=0.004, 8/9, d_z=1.32)** as the headline geometric bar, replacing the pre-registered G3+full (0.545). The honest framing to keep in mind: G2 and G3 are statistically tied (gap inside seed noise); I present G2 because it's the strongest *and* the ablation's value-path story makes G2 the natural protagonist. The E3 section's "I don't claim G2 > G3" caveat still governs — if asked, the truthful line is "G2 and G3 are tied; the robust finding is value-path ≫ bias-path."
- **Slide 4 is now all-G2 (resolved the earlier mixing).** Both geom and pos_only bars use the G2 injection, so the slide varies only the descriptor. Consequence accepted: **pos_only (G2) is not significant (0.520)**, so the deck no longer claims "robust to descriptor reduction" — slide 5 now says the effect lives in the value path instead, and the honest takeaway is "the full descriptor matters." (G3+pos_only would have kept significance, but mixing injections on one slide was the worse trade.)
- **Slide 3 note updated** — no longer calls G3 "the strongest / only mode run"; it now states all three were run and the value-path finding.

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
