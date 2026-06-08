# Geometrically Inductive Self-Supervised Learning for EEG: A Design Study of Spatial Attention Across Sessions, Subjects, and Montages

**Tianxin Zhou** — ECEN 525, Brain–Computer Interaction, Santa Clara University, Spring 2026.

**Project repository:** [https://github.com/tianxin-scu/geometric-eeg-ssl](https://github.com/tianxin-scu/geometric-eeg-ssl)

---

## Abstract

I propose, implement, and evaluate a self-supervised EEG encoder whose spatial attention is conditioned on a geometric descriptor $g_{ij}$ — built from the 3-D scalp positions of each electrode pair — instead of a learned per-electrode codex. The motivation is that EEG non-stationarity — across sessions, subjects, and montages — is largely a story of electrode geometry shifting, and a codex of per-electrode embeddings cannot absorb that shift smoothly because its parameters are indexed by electrode identity rather than by structure. A codex of per-electrode embeddings can be carried to a new montage only by reusing the rows of electrodes whose channel names recur; it has no embedding for genuinely unseen electrodes, and even matched-name rows are tuned to a different layout. The training recipe (momentum-encoder alignment + masked-patch reconstruction) follows EEGPT [Wang et al., 2024]. This report covers a leave-one-montage-out, mixed-corpus pretraining design that directly tests zero-shot cross-montage transfer — the property the proposal flagged as the central promise of the geometric approach. The primary baseline that can also transfer zero-shot is a channel-independent encoder (`chind`), which discards spatial structure; I additionally evaluate the transductive codex via name-aligned transfer (reusing matched-name embeddings), where it reaches only 0.522 BAC — above `chind` but below both geometric variants. The headline finding is positive: when pretrained on a balanced two-montage corpus of 156 subjects (PhysioNet MI + Sleep-EDFx) and evaluated zero-shot on the held-out 3-channel BCIC-2B montage, the geometric encoder reaches 0.545 BAC and beats the channel-independent baseline (0.509) by a statistically significant +0.036 (paired *t*-test $p=0.023$, Wilcoxon $p=0.027$, 7/9 subjects); the reduced position-only descriptor also beats the baseline significantly (+0.022, $p=0.013$, 8/9 subjects). At a smaller balanced 9-subject-per-dataset scale, the geometric encoder leads `chind` on all three held-out montages but the gaps are within noise. This is a design study: geometric conditioning delivers a significant zero-shot transfer advantage over the comparable baseline once the pretraining corpus is large enough, on the one held-out montage that the three available datasets allow to be tested at scale.

---

## 1. Introduction

### 1.1 Non-invasive BCI and EEG

Brain–computer interfaces (BCIs) offer transformative potential for individuals with motor impairments. Invasive modalities such as ECoG and Utah arrays deliver the highest signal fidelity but require neurosurgery and are inaccessible to most candidates. Electroencephalography (EEG) is the natural non-invasive alternative — safe, affordable, widely deployed, and capable of millisecond-scale recording — but its scalp signals are noisy, mixing many cortical sources through volume conduction and contaminated by muscle and ocular artifacts. Better representation learning on EEG has a direct ethical payoff: improved models extract more usable signal from the non-invasive modality, lowering the threshold at which a patient needs to accept surgical risk to achieve functional BCI performance.

### 1.2 EEG is non-stationary, by default

Any deployed EEG model faces a signal that shifts continuously along three axes, none of which is an experimental edge case:

1. **Cross-session drift.** Recordings of the same subject on different days differ because the cap sits differently each application (several-millimeter placement shifts are typical) and because the subject's physiological state varies (alertness, fatigue, electrode impedance).
2. **Cross-subject anatomical variability.** Head size, cortical folding, and skull thickness differ across individuals, so `C3` on one subject does not record from the same generators as `C3` on another.
3. **Cross-montage layout differences.** Clinical and consumer devices deploy 2 to 256 electrodes in heterogeneous layouts; a model trained on a 64-channel research montage has no principled way to process a 3-channel headset without retraining.

The placement, anatomical, and montage components of these shifts share a common structure: the geometry of the electrode array, relative to the underlying cortical sources, has changed. A model whose spatial representation is conditioned on geometry should absorb that change smoothly; a model whose spatial structure is encoded in per-electrode parameters cannot. Physiological state is a separate non-geometric axis of session variability and is not addressed by the architectural change proposed here.

### 1.3 The shared blind spot of EEG foundation models

Recent EEG foundation models — BENDR [Kostas et al., 2021], BIOT [Yang et al., 2023], LaBraM [Jiang et al., 2024], EEGPT [Wang et al., 2024] — bring self-supervised pretraining and linear-probe evaluation to EEG with strong downstream results, but they fall into two camps that both fail to handle EEG's natural non-stationarity. The first camp uses a learned **codex**: a table of per-electrode embeddings $\{c_i\}$ indexed by electrode identity, added to patch projections before any spatial mixing (EEGPT, LaBraM). The codex embedding for `C3` is a single vector fit to the average `C3` placement across training subjects; it cannot express that today's cap has shifted 5 mm or that this subject's head is larger than average, and an electrode whose name never appeared in pretraining has no codex entry at all — so such a montage cannot be processed without inventing the missing rows. This is the classical **transductive** setting [Hamilton et al., 2017]: parameters indexed by node identity, no generalization to unseen nodes without retraining. The second camp removes per-electrode structure entirely: BIOT tokenizes each channel independently, robust to variable channel counts but discarding the 3-D scalp coordinates that every EEG system provides as side information. Neither camp uses geometry as an input.

### 1.4 Contributions

I implement and evaluate the architecture committed to in the project proposal, and I focus the evaluation squarely on the cross-montage transfer the proposal called the headline test. Concretely:

- **An architecture** in which per-electrode codex embeddings are replaced by a shared function over $g_{ij}$ inside the attention computation, with no per-electrode parameters and an optional masked-reconstruction head — making the encoder *inductive* over electrode sets, and therefore runnable on any montage with known coordinates.
- **A leave-one-montage-out, mixed-corpus pretraining protocol** that trains on two datasets and evaluates zero-shot on the held-out third, isolating cross-montage generalization as the dependent variable.
- **Two zero-shot-capable baselines.** The channel-independent encoder (`chind`), trained under identical conditions, is the primary paired-significance comparison. I additionally evaluate the transductive codex by **name-aligned transfer** — reusing the codex rows of held-out electrodes whose names recur in pretraining; it reaches only 0.522 BAC, below both geometric variants, even though all of the held-out montage's electrodes matched (its best case).
- **A descriptor ablation.** I fix the injection to G3 (geometry at both attention score and value, the strongest configuration in a within-family injection ablation) and ablate the descriptor content: the full $g_{ij}$ vs. a position-only variant, asking whether absolute electrode positions alone carry the cross-montage signal.
- **The headline result.** At sufficient pretraining scale the geometric encoder beats `chind` on zero-shot transfer to the held-out montage by a statistically significant margin; at small scale the advantage is directionally consistent across all three held-out montages but not individually significant.

---

## 2. Problem Statement

### 2.1 Formal setup

Let $\mathbf{X} \in \mathbb{R}^{M \times T}$ denote a multichannel EEG recording with $M$ electrodes and $T$ time samples, and let $\mathbf{p}_i \in \mathbb{R}^3$ denote the known 3-D scalp position of electrode $i$, taken from the recording montage's digitized coordinates or from a standard template when digitization is unavailable. Self-supervised pretraining learns an encoder $f_\theta : \mathbb{R}^{M \times T} \to \mathbb{R}^d$ from unlabeled recordings; pretraining succeeds if a linear classifier trained on top of $f_\theta$ transfers across diverse downstream tasks. This linear-probing protocol is the standard evaluation framework in EEG foundation-model work and I adopt it here.

### 2.2 What this project isolates

The open design choice I focus on is how the spatial encoder represents inter-electrode relationships. Existing foundation models are split between codex-based spatial encoders (transductive in electrode identity) and channel-independent encoders (no spatial structure at all). I propose a third option: a spatial attention function conditioned on a geometric descriptor $g_{ij}$ (§4.2) built from electrode coordinates, with no per-electrode parameters, so that the same shared function applies to any electrode pair given only their 3-D positions — making the encoder inductive over electrode sets in the sense of [Hamilton et al., 2017]. The practical consequence this report tests is direct: such an encoder can be applied, frozen, to a montage it never saw in pretraining.

### 2.3 Research question

> Does conditioning the spatial encoder's attention on electrode geometry — rather than on a learned codex of electrode identities — produce representations that transfer zero-shot to an unseen electrode montage, beating the only baseline that can also transfer (a channel-independent encoder)?

This is a design study, not a SOTA competition. I do not ask whether my model outperforms LaBraM or EEGPT trained at scale; I ask whether, controlling for parameter count and pretraining data, geometric conditioning provides a cross-montage transfer advantage over the comparable baseline — and whether that advantage is large enough to be statistically real.

---

## 3. Related Work

### 3.1 EEG foundation models

Recent EEG foundation models share a common pipeline: large-scale self-supervised pretraining followed by linear probing or light fine-tuning. BENDR [Kostas et al., 2021], BIOT [Yang et al., 2023], LaBraM [Jiang et al., 2024], and EEGPT [Wang et al., 2024] collectively show that this recipe transfers across motor-imagery and clinical EEG tasks. A shared limitation remains: codex-based models encode space through per-electrode identities (transductive), while channel-independent models omit spatial structure entirely. Montage heterogeneity — 2 to 256 channels in varying layouts — is the deployment bottleneck this work addresses.

### 3.2 Self-supervised learning with momentum encoders

Momentum-encoder learning from MoCo [He et al., 2020] and BYOL [Grill et al., 2020] provides stable targets without requiring hard negative mining, which is especially attractive in EEG. MAE [He et al., 2022] adds a reconstruction objective that encourages representations to preserve recoverable signal structure. I adopt this validated combination (alignment + optional reconstruction), as also demonstrated in EEGPT, and focus my novelty on the spatial representation rather than inventing a new objective.

### 3.3 Inductive representations and attention over relational structure

The framing of this work — replacing identity-indexed parameters with a shared function over structural inputs — inherits its vocabulary from the inductive node-embedding literature. GraphSAGE [Hamilton et al., 2017] drew the explicit distinction between *transductive* embeddings (fit per-node, fixed graph) and *inductive* aggregation functions (shared parameters over node and neighbourhood features), generalizing to nodes unseen during training; that distinction is exactly the one I draw between codex-based and geometry-conditioned spatial encoders. GAT [Veličković et al., 2018] showed that attention is a natural way to implement such an aggregator, with attention scores conditioned on relational features rather than fixed by graph topology — the design pattern I adopt inside the spatial Transformer. EEG connectivity studies independently document that task-relevant information is partly relational rather than purely per-channel [Bastos & Schoffelen, 2016], which is a separate motivation for letting inter-electrode relationships shape spatial computation.

This is a Transformer with a geometry-conditioned attention bias; it does not construct a sparse graph or perform message passing in the GNN sense, and full $M \times M$ attention is well within budget at the electrode counts considered ($M \leq 64$).

---

## 4. Proposed Solution

### 4.1 Architecture overview

```
                  raw EEG signal X in R^{M x T}
                             |
                  +----------+----------+
                  |  patching (250 ms)  |
                  +----------+----------+
                             |
                       patches p_{i,t}
                             |
                  +----------+----------+
                  |  linear projection  |
                  +----------+----------+
                             |
                        x_{i,t} in R^d
                             |
              (NO codex embedding in geometric variant)
                             |
                             v
        +-------------------------------------------+
        |   GEOMETRIC SPATIAL TRANSFORMER (online)  |
        |   self-attention conditioned on g_{ij}    |
        |   no per-electrode parameters             |
        +-------------------------------------------+
                             |
                      S_t (CLS token)
                             |
                             v
                ----- alignment loss ----->  S_t (momentum encoder)
                                            (EMA tau cosine 0.996 -> 1.0)
                             |
                             v
                +-----------------------+
                | TEMPORAL TRANSFORMER  |
                | (over t = 1..T_p)     |
                +-----------------------+
                             |
                             v
                optional reconstruction head
                (masked patch prediction)
```

The single architectural choice that distinguishes this design from existing EEG foundation models lives inside the spatial encoder: how electrode-specific information enters the attention computation. All other components — patching, momentum updates, masking, temporal contextualization, and loss — follow standard practice [Grill et al., 2020; He et al., 2022; Wang et al., 2024]. The backbone (`src/geo/model/backbone.py`) recomputes geometry tables per dataset per step, because the montage — and therefore $g_{ij}$ — changes between corpora within a single mixed-corpus pretraining run.

### 4.2 The geometric descriptor

For each ordered pair of electrodes $(i, j)$ with 3-D scalp coordinates $p_i, p_j \in \mathbb{R}^3$, the headline descriptor is

$$g_{ij} = \big[\,p_i,\; p_j,\; p_i - p_j,\; \|p_i - p_j\|_2\,\big] \in \mathbb{R}^{10}. \qquad (1)$$

it combines absolute positions ($p_i, p_j$), signed displacement ($p_i - p_j$, distinguishing anterior/posterior and left/right), and distance (a scalar proximity term). Adding the two absolute positions to a displacement-and-distance descriptor lets the shared MLP localize a pair on the scalp, not merely relate the two electrodes. The descriptor is asymmetric in $(i, j)$, so attention can express directional preferences. Because $g_{ij}$ is a function of coordinates alone and contains no electrode-identity parameters, the same descriptor space is defined for any montage with known scalp coordinates — including montages absent from pretraining.

Coordinates are computed in a fixed scale from each dataset's channel names via `src/geo/preprocess.ch_pos_from_names`, with no per-montage normalization. This matters for cross-montage consistency: the encoder must see the held-out montage's coordinates on the same scale it saw during pretraining, or the transfer is confounded.

**Descriptor ablation (pos_only).** I also evaluate a reduced descriptor $g_{ij} = [\,p_i, p_j\,] \in \mathbb{R}^6$ (absolute positions only, dropping the displacement and distance terms). This asks whether absolute position alone carries the cross-montage signal, or whether the relational displacement/distance terms are doing the work.

### 4.3 Geometric vs. channel-independent spatial attention; injection variants

Writing the standard dot-product attention as $\alpha_{ij} \propto \exp((W_Q x_i)^\top (W_K x_j) / \sqrt{d})$ with output $y_i = \sum_j \alpha_{ij} W_V x_j$, geometry can enter at the score, the value, or both:

- **G1 (score only).** An additive per-head geometric bias, $e_{ij} = (W_Q x_i)^\top (W_K x_j)/\sqrt{d_h} + b_h(g_{ij})$, where $b_h$ is a shared MLP $\mathbb{R}^{10} \to \mathbb{R}^H$. Geometry conditions which electrodes attend.
- **G2 (value only).** Standard score; geometry-augmented value $v_{j \to i} = W_V x_j + W_{V,g}\, g_{ij}$. Geometry conditions the content of each message rather than the attention weights.
- **G3 (both).** G1 score-side bias *and* G2 value-side addition, with separate projections of the same $g_{ij}$.

Because G3 was consistently the strongest of the three injection points in a within-family ablation, I fix the injection to G3 and instead ablate the descriptor content (§4.2). The channel-independent baseline (`chind`) is the same backbone with `geometry_injection='none'`: standard dot-product attention with no geometric term and no per-electrode parameters. Because `chind` carries no electrode-identity parameters either, it too can be applied to an unseen montage — which is exactly why it is the right baseline for a zero-shot cross-montage test.

### 4.4 Self-supervised objective

The encoder is supervised by the combination of momentum-encoder alignment and masked-patch reconstruction:

$$\mathcal{L} = \mathcal{L}_A + \lambda_R \mathcal{L}_R. \qquad (2)$$

- $\mathcal{L}_A$: alignment loss, $-2 \cdot \cos(\text{pred}(z_\text{online}),\,\text{stop\_grad}(z_\text{momentum}))$ averaged over batch and time.
- $\mathcal{L}_R$: masked-patch reconstruction MSE, applied only at masked time steps; the reconstruction target is per-channel.
- Default $\lambda_R = 1$ with 50 % temporal masking (temporal-only; no spatial masking).

The momentum encoder is a non-trainable EMA copy of the online encoder, processing the unmasked input. The EMA factor follows a cosine ramp $\tau(k) = 1 - (1 - \tau_0) \frac{\cos(\pi k/K) + 1}{2}$ with $\tau_0 = 0.996$ and $\tau_K = 1.0$, matching the EEGPT recipe.

### 4.5 Baselines: channel-independent and name-aligned codex

The primary comparison baseline is the **channel-independent** encoder of §4.3, trained under identical conditions and — like the geometric encoder — genuinely montage-agnostic, so it transfers zero-shot without any fallback. I additionally evaluate the **transductive codex** cross-montage by **name-aligned transfer**: for each held-out electrode whose channel name recurs in the pretraining montage, the corresponding source codex row is reused; unmatched electrodes are randomly initialized. Unlike a spatially-nearest-neighbour fallback, name alignment introduces no geometry at transfer time — it reuses only identity-indexed embeddings, so the contrast with the geometric encoder stays clean. For BCIC-2B all three channels (C3/Cz/C4) matched, so the codex received its full trained embeddings (its best case). The codex checkpoint was pretrained on PhysioNet alone rather than the mixed corpus, a regime that favors it (more in-domain data, a fully matched montage); its underperformance against geometry (§7) is therefore conservative.

### 4.6 Why this design should help

EEG non-stationarity — across sessions, subjects, and montages — is largely a story of electrode geometry shifting. A model whose spatial attention is a function of geometry absorbs that shift smoothly, while one whose spatial structure lives in identity-indexed parameters can at best reuse embeddings for electrodes whose names recur, and cannot generalize past them. Whether this advantage appears empirically, and whether it survives a paired significance test, is what the experiments answer (§7).

---

## 5. Experiments

The experiment is a single design: leave-one-montage-out, mixed-corpus pretraining, with zero-shot linear-probe evaluation on the held-out montage. This isolates cross-montage transfer — the proposal's headline promise — as the dependent variable, and drops the in-distribution and cross-session evaluation rungs (which showed no separation).

**Leave-one-out splits.** Three pretraining runs per variant, each holding out one dataset:

| Tag | Pretrain on | Held out (zero-shot eval) |
|---|---|---|
| `no_sleep` | PhysioNet MI + BCIC-2B | Sleep-EDFx |
| `no_bcic`  | PhysioNet MI + Sleep-EDFx | BCIC-2B |
| `no_phys`  | BCIC-2B + Sleep-EDFx | PhysioNet MI |

**Variants.** `geometric` (G3, full 10-D $g_{ij}$), `geometric_pos_only` (G3, 6-D position-only descriptor), and `chind` (no geometry). Every variant is trained under identical conditions — same architecture, optimizer, schedule, masking, loss, and EMA schedule — so only the spatial encoder differs.

**Two scales.**

1. **Balanced n9.** Subjects capped at 9 per dataset (BCIC-2B has only 9 total — that hard ceiling sets the shared budget, so every dataset contributes equal subject-level diversity). All three splits × three variants → the 3×3 directional table (§7.1).
2. **Scaled n156.** The `no_bcic` split scaled to 78 subjects each from PhysioNet MI and Sleep-EDFx (a balanced 156-subject corpus), evaluated zero-shot on BCIC-2B. This is the only split that can be both balanced and large: BCIC-2B caps any corpus that contains it at 9, so the one large balanced corpus is the one that holds BCIC out. This run is the powered headline test (§7.2).

The pipeline is three scripts: `scripts/pretrain.py` (mixed-corpus pretraining per `(variant, split)`), `scripts/probe.py` (frozen-backbone zero-shot linear probe on the held-out montage), and `scripts/results.py` (aggregates the per-checkpoint JSONs into the headline tables under `results/n9/` and `results/n78/`).

---

## 6. Experimental Setup and Datasets

### 6.1 Datasets

I use three open EEG datasets accessible through MNE-Python [Gramfort et al., 2013] and MOABB [Aristimunha et al., 2023]:

| Dataset     | Paradigm                 | Subjects (available) | Ch.  | Native rate | Role |
| ----------- | ------------------------ | -------- | ---- | ----------- | ---------------- |
| PhysioNet MI [Schalk et al., 2004] | Motor imagery (4-class) | 105 | 64 | 160 Hz | pretrain / held-out (`no_phys`) |
| BCIC-2B [Tangermann et al., 2012] | Motor imagery (binary)  | 9  | 3 (C3,Cz,C4)  | 250 Hz | pretrain / **held-out (`no_bcic`)** |
| Sleep-EDFx [Kemp et al., 2000] | Sleep staging | 78 | 2 bipolar | 100 Hz | pretrain / held-out (`no_sleep`) |

Four PhysioNet MI subjects (88, 92, 100, 104) are excluded for known annotation/recording inconsistencies, leaving 105 usable.

**The 9-subject ceiling, and why it shapes the whole design.** BCIC-2B has only 9 subjects total. This single fact determines what experiments are possible: (a) any *balanced* pretraining corpus that includes BCIC-2B is capped at 9 subjects per dataset; (b) therefore the only corpus that can be both balanced and large (n156) is the one that *holds BCIC-2B out* — the `no_bcic` split; and (c) BCIC-2B's 3-channel layout is the most distinct montage of the three (vs. 64-channel PhysioNet MI and 2-channel Sleep-EDFx), which makes it the most informative zero-shot transfer target. The 9-subject ceiling is thus simultaneously the reason the powered test is possible and the reason there is only one of it.

**BCIC-2A from the proposal is not used.** 2A is 22-channel 4-class motor imagery on 9 subjects — a third in-distribution-style motor-imagery dataset rather than a new perturbation axis, and equally subject-capped. It is reserved as future work (§11).

### 6.2 Preprocessing (Tier 1, locked across datasets)

Identical preprocessing is applied to all three datasets so cross-dataset comparisons isolate dataset characteristics, not preprocessing decisions:

- **Bandpass** 0.5–45 Hz, zero-phase FIR.
- **Resample** to 200 Hz (PhysioNet 160 → up, BCIC 250 → down, Sleep-EDFx 100 → up; all end identical).
- **Epoch length** 4 s (16 patches at 250 ms per epoch).
- Per-epoch linear detrend, per-epoch per-channel z-score.
- No baseline correction (SSL has no privileged pre-cue interval), no re-referencing, no ICA.
- **Montage** `standard_1005`; **fixed-scale coordinates** (no per-montage normalization — §4.2).

**Sleep-EDFx specifics.** The dataset uses two bipolar derivations (`Fpz–Cz`, `Pz–Oz`) not in any standard montage; I place each bipolar electrode at the midpoint of its reference pair in `standard_1005` coordinates. Sleep-EDFx uses 30-second sleep-stage windows; I slice each into seven non-overlapping 4 s sub-epochs inheriting the parent label, to keep the backbone unchanged.

**Tier 3 artifact policy** (asymmetric). Pretraining clips to ±20σ after z-score (catches catastrophic artifacts, leaves typical EOG/EMG for the model to learn to ignore). Evaluation applies additional amplitude rejection.

### 6.3 Architecture defaults

- Token dimension $d = 256$; spatial depth $L_S = 2$; temporal depth $L_T = 4$; 8 heads; MLP ratio 4.
- Patch length 50 samples (250 ms at 200 Hz), non-overlapping; 16 patches per 4 s epoch.
- Geometric MLP hidden width 32. Temporal positional embedding: learned.

The geometric and channel-independent backbones share an identical temporal-Transformer / projection / patcher stack (~4.8 M online parameters in the backbone). Only the spatial encoder differs: the geometric variants add a small shared MLP over $g_{ij}$ (no per-electrode parameters); `chind` adds nothing. Because neither variant carries per-electrode parameters, both are montage-agnostic and can be evaluated zero-shot.

### 6.4 Training

- **Optimizer:** AdamW, learning rate 1e-4, weight decay 0.01.
- **Schedule:** linear warmup for 10 epochs, then cosine decay over 100 total epochs.
- **EMA τ:** cosine ramp $\tau_0 = 0.996 \to \tau_K = 1.0$.
- **Batch size:** 64. **Loss:** $\lambda_R = 1$, temporal masking at 50 %.
- **Epoch budget:** 4096 epochs per dataset per pass (small datasets oversampled, large ones subsampled to this exact count, equalizing gradient weight across the mixed corpus).

Geometry tables are recomputed each training step (the geometric MLPs are trainable, and the montage changes between corpora within a run); at inference there is no backward pass, so tables are precomputed once per montage.

### 6.5 Compute

Pretraining runs on Google Colab (T4 / A100) and on a local NVIDIA RTX 4070. A balanced n156 pretraining run (100 epochs) completes in ~40 min on the RTX 4070; the zero-shot probe adds ~2 min. Every checkpoint is paired with a `config.yaml` (resolved Tier 1 / ablation / Tier 3 / training settings) and a `run.txt` recording variant, pretrain datasets, held-out dataset, epoch budget, and subject count — so evaluation never relies on source-code defaults.

### 6.6 Linear-probe protocol

For each `(variant, split)` checkpoint:

1. Load the pretrained online backbone, freeze all parameters.
2. Recompute the held-out montage's fixed-scale `ch_pos` from its channel names (§4.2).
3. Extract a fixed feature per epoch: mean-pool the backbone output over electrodes and time → a single $\mathbb{R}^{256}$ vector.
4. Fit `StandardScaler` on the training split only, then a strictly linear `LogisticRegression` (no MLP probe), so the result reflects representation quality, not probe capacity.
5. Report Balanced Accuracy (primary), Cohen's κ, and weighted F1, per subject and aggregated.

Per held-out dataset (9 subjects each for the balanced table, for cross-row comparability):

- **PhysioNet MI / BCIC-2B:** leave-one-subject-out (LOSO).
- **Sleep-EDFx:** within-subject night split (train on night 1, test on night 2; aggregate over subjects with both nights).

The whole held-out dataset is excluded from pretraining, so every evaluated subject is genuinely zero-shot.

### 6.7 Significance testing

For the headline n156 → BCIC test I run a **paired** test on per-subject BAC differences between a geometric variant and `chind` over the 9 LOSO folds (the same held-out subjects for both models). I report both a paired $t$-test and a Wilcoxon signed-rank test (`scipy.stats`) at $\alpha = 0.05$, plus the paired effect size $d_z$ and the sign count.

---

## 7. Results

All BAC values are LOSO or within-subject-night means ± standard deviations across the evaluated subjects. Source files: `results/n9/headline.txt` (balanced n9) and `results/n78/headline.txt` (scaled n156).

### 7.1 Balanced n9 — directional table across all three held-out montages

| Held-out (eval)            | geometric (full) | geometric (pos_only) | chind |
| -------------------------- | ---------------- | -------------------- | ----- |
| Sleep-EDFx (night split)   | 0.593 ± 0.022    | 0.597 ± 0.022        | 0.589 ± 0.020 |
| BCIC-2B (LOSO)             | 0.509 ± 0.026    | 0.508 ± 0.019        | 0.504 ± 0.024 |
| PhysioNet MI (LOSO)        | 0.303 ± 0.048    | 0.315 ± 0.044        | 0.302 ± 0.047 |

**Reading.** With a balanced 9-subject-per-dataset corpus, both geometric variants beat `chind` on all three held-out montages. The direction is consistent, but the gaps are small (≤1.3 BAC points) and within the per-subject standard deviation, so no single row is significant on its own. The three rows are different tasks with different chance levels (BCIC-2B binary, chance 0.50; PhysioNet MI 4-class, chance 0.25; Sleep-EDFx multi-class), so the comparison is within each row (geometric vs. chind), not across rows. This table establishes a consistent directional signal and motivates scaling the one split that can be scaled.

### 7.2 Scaled n156 — the powered zero-shot transfer test (held-out BCIC-2B)

Pretraining on a balanced 156-subject corpus (PhysioNet MI 78 + Sleep-EDFx 78), evaluated zero-shot on BCIC-2B (9 LOSO folds):

| Variant                    | BCIC-2B zero-shot BAC | n (folds) |
| -------------------------- | --------------------- | --------- |
| chind (baseline)           | 0.509 ± 0.016         | 9 |
| codex (name-aligned)       | 0.522 ± 0.021         | 9 |
| geometric (pos_only)       | 0.531 ± 0.013         | 9 |
| **geometric (full $g_{ij}$)** | **0.545 ± 0.027**  | 9 |

**Paired tests vs. `chind`** over the same 9 held-out subjects (each model evaluated on the identical LOSO folds), both significant at $\alpha = 0.05$ on the parametric and non-parametric test:

| Comparison | mean diff | wins | paired $t$ | Wilcoxon | $d_z$ | 95 % CI |
| --- | --- | --- | --- | --- | --- | --- |
| geometric (full) − chind | **+0.036** | 7/9 | $p = 0.023$ | $p = 0.027$ | 0.93 | $[+0.006, +0.066]$ |
| geometric (pos_only) − chind | **+0.022** | 8/9 | $p = 0.013$ | $p = 0.012$ | 1.07 | $[+0.006, +0.037]$ |

Both confidence intervals exclude zero.

**Reading.** When the pretraining corpus is large enough, the geometric advantage over the baseline becomes both larger and statistically significant for both descriptors. The full 10-D descriptor gives the strongest point estimate (0.545, +0.036 over `chind`); the reduced position-only descriptor still beats `chind` significantly (+0.022), which shows the cross-montage signal is robust even to dropping the displacement and distance terms. For contrast, the same `no_bcic` comparison at the balanced n9 scale (§7.1) showed only +0.003 BAC (3/9 subjects, not significant) — the effect emerges with pretraining scale. The codex baseline (0.522, name-aligned, all three channels matched) lands above `chind` but below both geometric variants; because it comes from a different (PhysioNet-only) pretraining regime that favors it, no paired test is run against it — it is a deliberately conservative point-estimate comparison.

### 7.3 Headline summary

In the balanced small-corpus regime, geometric attention leads the channel-independent baseline on every held-out montage but within noise. On the one split that can be scaled to a large balanced corpus, geometric attention beats the baseline on zero-shot transfer to a never-seen 3-channel montage by a statistically significant margin. The transductive codex, evaluated by name-aligned transfer, sits between `chind` and the geometric variants (0.522) despite a regime that favors it — it can reuse embeddings for recurring electrode names but cannot generalize beyond them.

---

## 8. Discussion

### 8.1 What the results say

The experiments support the proposal's central claim on the test they can support best. Geometric conditioning produces representations that transfer zero-shot to an unseen montage and beat the channel-independent baseline — the only one that transfers without modification — by a margin that survives a paired significance test once the pretraining corpus is large enough (n156). They also beat the name-aligned codex, whose best case still trails both geometric variants. At small balanced scale (n9) the advantage is consistent in direction across all three held-out montages but not in magnitude — the gaps sit inside the per-subject noise.

### 8.2 Why the effect emerges with scale

The contrast between the n9 `no_bcic` gap (+0.003, not significant) and the n156 `no_bcic` gap (+0.022, $p=0.013$) is the most informative comparison in the project. Two readings, both consistent with the data: (1) inductive biases help most once the corpus is large enough that an unstructured baseline would otherwise need to fit montage-specific structure it cannot share — the geometric encoder shares that structure through $g_{ij}$ and so generalizes; and (2) 18 subjects is too few to resolve a ~2-point BAC difference against ~2-point per-subject variance, whereas a larger, more diverse corpus both grows the gap and tightens the estimate. The descriptor ablation supports the first reading: even the position-only descriptor retains a significant advantage, so the signal is carried by geometry broadly rather than by one term.

### 8.3 The codex baseline, and why geometry still wins

The transductive codex of foundation-model papers can be carried to an unseen montage by name-aligned transfer — reusing the trained embedding of every held-out electrode whose name recurs in pretraining. For BCIC-2B that covers all three channels, so the codex received its full trained embeddings, from a PhysioNet-only regime that favors it; it still reached only 0.522, below both geometric variants. Name alignment, unlike a nearest-neighbour spatial fallback, injects no geometry at transfer time, so this stays a clean transductive-vs-inductive contrast: the codex generalizes only to electrode names it has already seen, while the geometric encoder generalizes to any coordinates. The channel-independent encoder remains the primary paired baseline because it is montage-agnostic by construction; the geometric encoder beats both.

### 8.4 Relation to the proposal hypothesis

The proposal expected geometric conditioning to help most in cross-montage transfer. In this regime that expectation holds, with the important qualifier that it holds significantly on one held-out montage (BCIC-2B) at scale, and directionally on the other two at the small balanced scale they permit. The mechanism works on the test designed to expose it, and the bound is one of data availability (§10) rather than a contradiction of the hypothesis.

---

## 9. Conclusion

I implemented and evaluated a geometrically inductive self-supervised EEG encoder: spatial attention conditioned on a geometric descriptor $g_{ij}$ instead of a learned per-electrode codex. I tested the property the proposal called central — zero-shot transfer to an unseen montage — with a leave-one-montage-out, mixed-corpus protocol, comparing against a channel-independent encoder (the only baseline that transfers without modification) and a name-aligned transductive codex, on three open EEG datasets.

The headline result is positive: pretrained on a balanced 156-subject corpus and evaluated zero-shot on the held-out 3-channel BCIC-2B montage, the geometric encoder reaches 0.545 BAC and beats the channel-independent baseline (0.509) by a statistically significant +0.036 BAC ($p=0.023$, 7/9 subjects); the position-only descriptor also beats it significantly (+0.022, $p=0.013$, 8/9 subjects). The name-aligned codex, in its best case, reaches only 0.522 — below both geometric variants. At small balanced scale the advantage is directionally consistent across all three held-out montages but within noise. Framed as a design study, the project answers its research question — *does geometric conditioning transfer zero-shot to an unseen montage, better than the comparable baseline?* — with a measured **yes, at sufficient scale, on the montage the available data lets me test at scale**, and it names the specific follow-ups that would extend the answer to more montages.

---

## 10. Limitations

1. **One powered held-out montage.** The significant result is on BCIC-2B only. The other two held-out directions (Sleep-EDFx, PhysioNet MI) can only be tested at the balanced n9 scale, where the gaps are directional but underpowered — because any balanced corpus *containing* BCIC-2B is capped at 9 subjects. The single powered test is a genuine result, but it is one montage, not a montage sweep.
2. **Small evaluation cohort.** BCIC-2B has only 9 subjects, so the headline significance rests on 9 paired LOSO folds. The consistency (8/9 same-sign) and the agreement of the parametric and non-parametric tests guard against this, but per-subject BAC at n=9 is higher-variance than a full-population estimate; sub-2-point gaps should be read cautiously.
3. **Pretraining scale.** n156 is large for this project but small by foundation-model standards ($10^1$–$10^2$ subjects vs. $10^3$–$10^4$ hours). The geometric advantage grows from n9 to n156; this report cannot characterize where it saturates.
4. **Single seed for the headline.** Time and compute did not permit multi-seed averaging; per-subject variance is reported but per-seed variance is not.
5. **Descriptor still moderate.** The 10-D $g_{ij}$ captures absolute position, displacement, and distance, but not hemisphere indicators, geodesic distance on a cortical mesh, or anatomy. Richer descriptors are future work.
6. **Codex regime mismatch.** The codex baseline is evaluated by name-aligned transfer, but its checkpoint was pretrained on PhysioNet alone, not the mixed 156-subject corpus. The mismatch *favors* the codex, so the comparison is conservative; a mixed-corpus codex pretrain would make the codex bar apples-to-apples (§11). No paired significance test is run for the codex, as it shares no folds across regimes.
7. **Cross-session and in-distribution rungs dropped.** This study focuses on cross-montage transfer; the cross-session (Sleep-EDFx night-split) and in-distribution evaluation rungs are not run.

---

## 11. Future Work

The shape of the results points to concrete follow-ups, in roughly decreasing value:

1. **A second distinct-montage held-out test.** The cleanest way to turn the single-montage result into a generalization claim is to hold out a different distinct montage with enough subjects to be powered — e.g. a high-density (≈128-channel) motor-imagery dataset, or the 22-channel BCIC-2A — while training on a large, diverse corpus. This directly addresses Limitation 1 and is the single most informative next experiment. (New datasets require sign-off per the project's standing constraints; this is flagged as future work, not scoped into the present study.)
2. **Scale the corpus further and characterize the curve.** The advantage grew from n9 to n156; pretraining on a larger, more diverse multi-dataset corpus and re-running the held-out probe would show whether the gap keeps growing or saturates.
3. **Mixed-corpus codex pretrain.** Pretrain the transductive codex on the same mixed 156-subject corpus, then re-probe BCIC-2B with name-aligned transfer, to make the codex bar apples-to-apples with the geometric and `chind` runs (the present codex bar uses a PhysioNet-only checkpoint).
4. **Richer geometric descriptors with G3.** Push the fixed-G3 design with spherical coordinates, an explicit hemisphere indicator, or geodesic distance on a cortical-surface mesh, building on the descriptor ablation here (which shows even position-only already transfers).
5. **Multi-seed the headline** to put a confidence band on the n156 result, and re-add the cross-session and in-distribution rungs for completeness.

---

## 12. Contributions

This is a single-author project. All design, implementation, experiments, and writing are by Tianxin Zhou. No collaborators contributed code or text.

I used **GitHub Copilot / Claude Code** as a coding assistant during implementation: pairing on smoke-test scaffolds, refactors, and documentation. The architectural decisions, experimental design, dataset choices, and the analysis of the results in this report are mine. Where the assistant drafted prose, I rewrote it before inclusion. The course's policy that "using gen-AI solely to generate your project report ... will result in you failing the course" is the boundary I operated within: the engineering work, the experimental judgement, and the framing of the result are my own.

---

## 13. Reused Open-Source Code

- **EEGPT** (Wang et al., 2024) — [https://github.com/BINE022/EEGPT](https://github.com/BINE022/EEGPT). I read the EEGPT codebase to validate (a) the cosine ramp for the EMA $\tau$ schedule, (b) stop-gradient placement in the alignment loss, and (c) the masked-patch reconstruction convention. Apache 2.0; my code is an independent implementation, not a fork.
- **MNE-Python** [Gramfort et al., 2013] — data loading, montage handling, EDF parsing, bandpass filtering, resampling, Sleep-EDFx access.
- **MOABB** [Aristimunha et al., 2023] — BCIC-2B dataset access (`BNCI2014_004`); MOABB's dataset registry was also surveyed for future-work montage candidates.
- **PyTorch** — backbone, attention, training loop.
- **scikit-learn** — `StandardScaler`, `LogisticRegression`, balanced accuracy / Cohen's κ scoring.
- **SciPy** — `scipy.stats` paired $t$-test and Wilcoxon signed-rank test for the headline significance test.

The project itself is licensed under the LICENSE file in the public repository.

---

## 14. References

- Aristimunha, B. et al. (2023). *MOABB: trustworthy algorithm benchmarking for BCIs.* J. Neural Eng.
- Bastos, A. M. & Schoffelen, J.-M. (2016). *A tutorial review of functional connectivity analysis methods and their interpretational pitfalls.* Front. Syst. Neurosci.
- Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience.
- Grill, J.-B. et al. (2020). *Bootstrap your own latent (BYOL): a new approach to self-supervised learning.* NeurIPS.
- Hamilton, W. L., Ying, R. & Leskovec, J. (2017). *Inductive representation learning on large graphs (GraphSAGE).* NeurIPS.
- He, K. et al. (2020). *Momentum Contrast for unsupervised visual representation learning (MoCo).* CVPR.
- He, K. et al. (2022). *Masked autoencoders are scalable vision learners (MAE).* CVPR.
- Jiang, W. et al. (2024). *LaBraM: a large-scale brain-activity foundation model.* ICLR.
- Kemp, B. et al. (2000). *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG (Sleep-EDF).* IEEE Trans. Biomed. Eng.
- Kostas, D., Aroca-Ouellette, S. & Rudzicz, F. (2021). *BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data.* Front. Hum. Neurosci.
- Schalk, G. et al. (2004). *BCI2000: a general-purpose brain-computer interface (BCI) system.* IEEE Trans. Biomed. Eng. (PhysioNet MI dataset.)
- Tangermann, M. et al. (2012). *Review of the BCI Competition IV.* Front. Neurosci.
- Veličković, P. et al. (2018). *Graph Attention Networks (GAT).* ICLR.
- Wang, G. et al. (2024). *EEGPT: pretrained transformer for universal and reliable representation of EEG signals.* NeurIPS.
- Yang, C. et al. (2023). *BIOT: cross-data biosignal learning in the wild.* NeurIPS.
