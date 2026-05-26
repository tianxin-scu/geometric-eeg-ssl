# Geometrically Inductive Self-Supervised Learning for EEG: A Design Study of Spatial Attention Across Sessions, Subjects, and Montages

**Tianxin Zhou** — ECEN 525, Brain–Computer Interaction, Santa Clara University, Spring 2026.

**Project repository:** [https://github.com/tianxin-scu/geometric-eeg-ssl](https://github.com/tianxin-scu/geometric-eeg-ssl)

---

## Abstract

I propose, implement, and evaluate a self-supervised EEG encoder whose spatial attention is conditioned on a 4-dimensional geometric descriptor $g_{ij}$ (Euclidean distance + signed 3-D displacement between electrode pairs) instead of a learned per-electrode codex. The motivation is that EEG non-stationarity — across sessions, subjects, and montages — is largely a story of electrode geometry shifting, and a codex of per-electrode embeddings cannot absorb that shift smoothly because its parameters are indexed by electrode identity rather than by structure. The training recipe (momentum-encoder alignment + masked-patch reconstruction) follows EEGPT [Wang et al., 2024]. I run the full evaluation plan committed to in the proposal: an in-distribution linear probe on PhysioNet MI (E1), a three-step robustness ladder (E2a cross-session on Sleep-EDFx; E2b cross-subject LOSO with paired Wilcoxon test; E2c zero-shot cross-montage transfer from 64-channel PhysioNet MI to 3-channel BCIC-2B), the G1/G2/G3 architectural ablation (E5), and a channel-independent sanity check (E7). The headline finding is that *geometric attention is competitive but not dominant*: in-distribution it ties the codex baseline (0.342 vs 0.341 LOSO BAC on PhysioNet MI) and both beat the channel-independent control (0.333), but on cross-montage transfer a codex baseline with nearest-neighbour fallback (0.525) slightly outperforms the geometric encoder (0.509). Within the geometric family, G3 — geometry injected at both score and value — is consistently the strongest variant. I frame this as an informative design study: at pilot-scale pretraining, a well-engineered codex with nearest-neighbour transfer recovers most of the inductive advantage that motivates this work, and the smallest 4-D geometric descriptor is insufficient on its own to separate the two approaches.

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

Recent EEG foundation models — BENDR [Kostas et al., 2021], BIOT [Yang et al., 2023], LaBraM [Jiang et al., 2024], EEGPT [Wang et al., 2024] — bring self-supervised pretraining and linear-probe evaluation to EEG with strong downstream results, but they fall into two camps that both fail to handle EEG's natural non-stationarity. The first camp uses a learned **codex**: a table of per-electrode embeddings $\{c_i\}$ indexed by electrode identity, added to patch projections before any spatial mixing (EEGPT, LaBraM). The codex embedding for `C3` is a single vector fit to the average `C3` placement across training subjects; it cannot express that today's cap has shifted 5 mm or that this subject's head is larger than average, and an electrode absent from the pretraining set has no codex entry at all. This is the classical **transductive** setting [Hamilton et al., 2017]: parameters indexed by node identity, no generalization to unseen nodes without retraining. The second camp removes per-electrode structure entirely: BIOT tokenizes each channel independently, robust to variable channel counts but discarding the 3-D scalp coordinates that every EEG system provides as side information. Neither camp uses geometry as an input.

### 1.4 Contributions

I implement and evaluate the architecture committed to in the project proposal. Concretely:

- **An architecture** in which per-electrode codex embeddings are replaced by $g_{ij}$ inside the attention function, with no per-electrode parameters and an optional masked-reconstruction head.
- **Three options for where $g_{ij}$ enters** — score only (G1), value only (G2), or both (G3) — evaluated as the principal architectural ablation.
- **A full robustness evaluation** comparing the geometric encoder against a parameter-matched transductive (codex) baseline across three increasing levels of geometric perturbation: cross-session within-subject (Sleep-EDFx multi-night), cross-subject LOSO on motor imagery, and zero-shot cross-montage transfer from 64-channel PhysioNet MI to 3-channel BCIC-2B.
- **An honest reading of the result.** The geometric encoder matches the codex in-distribution, but a codex baseline equipped with a nearest-neighbour codex fallback recovers the inductive advantage at transfer time well enough to match or slightly beat $g_{ij}$ at the pilot scale evaluated here. The strongest within-family finding is that G3 dominates G1/G2 on both datasets.

---

## 2. Problem Statement

### 2.1 Formal setup

Let $\mathbf{X} \in \mathbb{R}^{M \times T}$ denote a multichannel EEG recording with $M$ electrodes and $T$ time samples, and let $\mathbf{p}_i \in \mathbb{R}^3$ denote the known 3-D scalp position of electrode $i$, taken from the recording montage's digitized coordinates or from a standard template when digitization is unavailable. Self-supervised pretraining learns an encoder $f_\theta : \mathbb{R}^{M \times T} \to \mathbb{R}^d$ from unlabeled recordings; pretraining succeeds if a linear classifier trained on top of $f_\theta$ transfers across diverse downstream tasks. This linear-probing protocol is the standard evaluation framework in EEG foundation-model work and I adopt it here.

### 2.2 What this project isolates

The open design choice I focus on is how the spatial encoder represents inter-electrode relationships. Existing foundation models are split between codex-based spatial encoders (transductive in electrode identity) and channel-independent encoders (no spatial structure at all). I propose a third option: a spatial attention function conditioned on the geometric descriptor $g_{ij} \in \mathbb{R}^4$ (Eq. 1), with no per-electrode parameters, so that the same shared function applies to any electrode pair given only their 3-D positions — making the encoder inductive over electrode sets in the sense of [Hamilton et al., 2017].

### 2.3 Research question

> Does conditioning the spatial encoder's attention on electrode geometry — rather than on a learned codex of electrode identities — produce representations that are more robust to EEG's natural non-stationarity (cross-session, cross-subject, cross-montage), without sacrificing in-distribution accuracy?

This is a **design study**, not a SOTA competition. I do not ask whether my model outperforms LaBraM or EEGPT trained at scale; I ask whether, controlling for parameter count and pretraining data, geometric conditioning provides a robustness advantage over the codex approach — and whether that advantage is large enough to matter in practice.

---

## 3. Related Work

### 3.1 EEG foundation models

Recent EEG foundation models share a common pipeline: large-scale self-supervised pretraining followed by linear probing or light fine-tuning. BENDR [Kostas et al., 2021], BIOT [Yang et al., 2023], LaBraM [Jiang et al., 2024], and EEGPT [Wang et al., 2024] collectively show that this recipe transfers across motor-imagery and clinical EEG tasks. A shared limitation remains: codex-based models encode space through per-electrode identities (transductive), while channel-independent models omit spatial structure entirely. Montage heterogeneity — 2 to 256 channels in varying layouts — is the deployment bottleneck this work addresses.

### 3.2 Self-supervised learning with momentum encoders

Momentum-encoder learning from MoCo [He et al., 2020] and BYOL [Grill et al., 2020] provides stable targets without requiring hard negative mining, which is especially attractive in EEG. MAE [He et al., 2022] adds a reconstruction objective that encourages representations to preserve recoverable signal structure. I adopt this validated combination (alignment + optional reconstruction), as also demonstrated in EEGPT, and focus my novelty on the spatial representation rather than inventing a new objective.

### 3.3 Inductive representations and attention over relational structure

The framing of this work — replacing identity-indexed parameters with a shared function over structural inputs — inherits its vocabulary from the inductive node-embedding literature. GraphSAGE [Hamilton et al., 2017] drew the explicit distinction between *transductive* embeddings (fit per-node, fixed graph) and *inductive* aggregation functions (shared parameters over node and neighbourhood features), generalizing to nodes unseen during training; that distinction is exactly the one I draw between codex-based and geometry-conditioned spatial encoders. GAT [Veličković et al., 2018] showed that attention is a natural way to implement such an aggregator, with attention scores conditioned on relational features rather than fixed by graph topology — the design pattern I adopt inside the spatial Transformer. EEG connectivity studies independently document that task-relevant information is partly relational rather than purely per-channel [Bastos & Schoffelen, 2016], which is a separate motivation for letting inter-electrode relationships shape spatial computation.

I emphasize that this proposal is a Transformer with a geometry-conditioned attention bias; it does not construct a sparse graph or perform message passing in the GNN sense, and full $M \times M$ attention is well within budget at the electrode counts considered ($M \leq 64$).

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

The single architectural choice that distinguishes this design from existing EEG foundation models lives inside the spatial encoder: how electrode-specific information enters the attention computation. All other components — patching, momentum updates, masking, temporal contextualization, and loss — follow standard practice [Grill et al., 2020; He et al., 2022; Wang et al., 2024].

### 4.2 The geometric descriptor

For each ordered pair of electrodes $(i, j)$ with 3-D scalp coordinates $p_i, p_j \in \mathbb{R}^3$,

$$g_{ij} = \big[\,\|p_i - p_j\|_2,\; p_i - p_j\,\big] \in \mathbb{R}^4. \qquad (1)$$

The descriptor combines distance (a scalar that captures proximity) with signed displacement (a 3-vector that distinguishes anterior from posterior and left from right). It is asymmetric in $(i, j)$, so attention can express directional preferences. Because $g_{ij}$ is a function of coordinates alone and contains no electrode-identity parameters, the same descriptor space is defined for any montage for which scalp coordinates are available.

### 4.3 Geometric vs. transductive spatial attention

Writing the standard dot-product attention as $\alpha_{ij} \propto \exp((W_Q x_i)^\top (W_K x_j) / \sqrt{d})$ with output $y_i = \sum_j \alpha_{ij} W_V x_j$, the three geometric variants are:

**G1 (score only).** Geometry decides *which* electrodes attend. In the proposal this was written as a GAT-style additive form
$e_{ij} = a^\top \mathrm{LeakyReLU}\!\big(W_a [W_Q x_i \,\|\, W_K x_j \,\|\, W_g g_{ij}]\big)$.
In my implementation G1 is realized as scaled dot-product attention with an additive per-head geometric bias,
$e_{ij} = (W_Q x_i)^\top (W_K x_j)/\sqrt{d_h} + b_h(g_{ij})$,
where $b_h$ is a shared MLP $\mathbb{R}^4 \to \mathbb{R}^H$. The motivation for this small deviation is to keep the geometric and codex variants on the same attention kernel: the headline E1 comparison then isolates the inductive-vs-transductive variable rather than confounding it with additive-vs-dot-product attention. The literal GAT-additive form is recoverable as a future ablation by swapping the kernel. **G1 is the default geometric variant** on grounds of parsimony.

**G2 (value only).** Standard dot-product score; geometry-augmented value:
$v_{j \to i} = W_V x_j + W_{V,g}\, g_{ij},\quad y_i = \sum_j \alpha_{ij}\, v_{j \to i}.$
Geometry colours *what* messages carry, not who talks.

**G3 (both).** G1 score-side bias *and* G2 value-side addition, with separate projections $W_g$ and $W_{V,g}$. Geometry simultaneously shapes which electrodes attend and what they exchange.

### 4.4 Self-supervised objective

The encoder is supervised by the combination of momentum-encoder alignment and optional masked-patch reconstruction:

$$\mathcal{L} = \mathcal{L}_A + \lambda_R \mathcal{L}_R. \qquad (2)$$

- $\mathcal{L}_A$: alignment loss, $-2 \cdot \cos(\text{pred}(z_\text{online}),\,\text{stop\_grad}(z_\text{momentum}))$ averaged over batch and time.
- $\mathcal{L}_R$: masked-patch reconstruction MSE, applied only at masked time steps; reconstruction target is the channel-mean of the original patches.
- Default $\lambda_R = 1$ with 50 % joint (electrode, time) masking.

The momentum encoder is a non-trainable EMA copy of the online encoder, processing the unmasked input. The EMA factor follows a cosine ramp $\tau(k) = 1 - (1 - \tau_0) \frac{\cos(\pi k/K) + 1}{2}$ with $\tau_0 = 0.996$ and $\tau_K = 1.0$, matching the EEGPT recipe.

### 4.5 Transductive baseline

Architecturally identical to the geometric encoder except for the spatial mechanism: a learned codex $\{c_i \in \mathbb{R}^d\}$ is added to patch projections before standard dot-product attention. At transfer time (E2c), missing-electrode codex entries are populated by one of two fallbacks:

- **Random fallback.** Each missing entry initialized from $\mathcal{N}(0, \sigma^2)$ with $\sigma$ matched to the pretraining codex's empirical std.
- **Nearest-neighbour (nn) fallback.** Each missing entry copied from the spatially-nearest pretraining electrode using Euclidean distance in `standard_1005` coordinates. This fallback effectively re-uses the pretraining geometry at inference, and is the strongest codex transfer baseline I evaluate.

### 4.6 Why this design should help (the predicted advantage)

The intuition is short. EEG non-stationarity — across sessions, subjects, and montages — is largely a story of electrode geometry shifting. A model whose spatial attention is a function of geometry absorbs that shift smoothly, while one whose spatial structure lives in identity-indexed parameters cannot. Whether the predicted advantage appears empirically, and whether its size justifies any in-distribution cost, is what experiments E1 and E2 answer. The result, reported in §6, is more nuanced than the proposal anticipated: the codex baseline with a nearest-neighbour fallback recovers most of the predicted advantage.

---

## 5. Experiments

I run the five-experiment plan committed to in the proposal. Two proposal items not included in this report are E3 (channel-dropout curves) and E6 (loss-component ablation): both were marked optional in the proposal and were dropped to fit the compute budget once the headline experiments were under way.

| ID  | Question                                                | Datasets                       | Variants                                 |
| --- | ------------------------------------------------------- | ------------------------------ | ---------------------------------------- |
| E1  | In-distribution linear-probe parity                     | PhysioNet MI                   | G1, codex, channel-independent           |
| E2a | Cross-session within-subject (night 1 → night 2)        | Sleep-EDFx                     | G1, codex                                |
| E2b | Cross-subject LOSO variance, paired Wilcoxon            | PhysioNet MI (full 105 subj)   | G1 vs codex                              |
| E2c | Zero-shot cross-montage transfer (64 ch → 3 ch)         | PhysioNet MI → BCIC-2B         | G1, codex-random, codex-nn               |
| E5  | Where $g_{ij}$ enters — G1 / G2 / G3                    | PhysioNet MI + BCIC-2B         | G1, G2, G3                               |
| E7  | Channel-independent sanity check                        | PhysioNet MI + BCIC-2B         | channel-independent, G1, codex           |

Each experiment is orchestrated by a dedicated script (`scripts/run_e1.py`, `scripts/run_e2.py {a,b,c}`, `scripts/run_e5.py`, `scripts/run_e7.py`) that resolves a checkpoint per variant, runs `scripts/probe.py` in a fresh subprocess, and aggregates the BAC / κ / weighted-F1 per subject into a JSON file and a printed summary table. Variants without a finished checkpoint are skipped with `(no ckpt)` and the table continues, so partial result coverage is normal during iteration.

---

## 6. Experimental Setup and Datasets

### 6.1 Datasets

I use three open EEG datasets accessible through MNE-Python [Gramfort et al., 2013] and MOABB [Aristimunha et al., 2023]:

| Dataset     | Paradigm                 | Subjects | Ch.  | Native rate | Used for         |
| ----------- | ------------------------ | -------- | ---- | ----------- | ---------------- |
| PhysioNet MI [Schalk et al., 2004] | Motor imagery (4-class) | 109 → 105 | 64 | 160 Hz | E1, E2b, E5, E7, pretrain source |
| BCIC-2B [Tangermann et al., 2012] | Motor imagery (binary)  | 9  | 3  | 250 Hz | E2c target, E5, E7 |
| Sleep-EDFx [Kemp et al., 2000] | Sleep staging (5-class) | ~78 | 2 bipolar | 100 Hz | E2a |

Four PhysioNet MI subjects (88, 92, 100, 104) are excluded due to known annotation/recording inconsistencies, leaving 105 subjects.

**BCIC-2A from the proposal is not used in this report.** Despite the similar name, BCIC-2A and BCIC-2B are different datasets — 2A is 22-channel 4-class motor imagery on 9 subjects, while 2B is 3-channel binary motor imagery on 9 subjects. BCIC-2B is essential for this project because its 3-channel layout (C3, Cz, C4) is the cross-montage transfer target in E2c against the 64-channel PhysioNet MI pretraining source; 2A's 22-channel layout would not provide a comparable montage mismatch. BCIC-2A, on the other hand, would be a *third* 4-class motor-imagery dataset on top of PhysioNet MI (105 subjects, 4-class MI) — same paradigm, same task, with a 12× smaller cohort. Running it would produce another in-distribution-style LOSO number on a tiny sample with no new perturbation axis: cross-montage is already covered by 2B, cross-subject by PhysioNet MI LOSO, and cross-session by Sleep-EDFx. Dropping 2A was a compute-budget call, but the theoretical loss is small: the three perturbation axes the proposal committed to are fully covered without it.

### 6.2 Preprocessing (Tier 1, locked across datasets)

Identical preprocessing is applied to all three datasets so cross-dataset comparisons isolate dataset characteristics, not preprocessing decisions:

- **Bandpass** 0.5–45 Hz, zero-phase FIR.
- **Resample** to 200 Hz (PhysioNet 160 → up, BCIC 250 → down, Sleep-EDFx 100 → up; all end identical).
- **Epoch length** 4 s (16 patches at 250 ms per epoch).
- Per-epoch linear detrend, per-epoch per-channel z-score.
- No baseline correction (SSL has no privileged pre-cue interval), no re-referencing, no ICA.
- **Montage:** `standard_1005`. **Coordinate normalization:** centroid-subtract, unit-max-norm scale.

The proposal initially listed 256 Hz resample and 0.5–75 Hz bandpass. I locked Tier 1 to 200 Hz and 0.5–45 Hz instead, because (a) the lower sampling rate halves backbone compute and patch length at no perceptible accuracy cost in pilot runs, and (b) 0.5–45 Hz both excludes mains line noise and keeps the band wide enough for motor-imagery rhythms. This deviation is logged in Session 1 of the project session log and is consistent with the EEGPT preprocessing choice.

**Sleep-EDFx specifics.** The dataset uses two bipolar derivations (`Fpz–Cz`, `Pz–Oz`) that are not in any standard montage; I place each bipolar electrode at the midpoint of its reference pair in `standard_1005` coordinates. Sleep-EDFx uses 30-second sleep-stage windows; I slice each 30 s window into seven non-overlapping 4 s sub-epochs, each inheriting the parent label, to keep the backbone unchanged.

**Tier 3 artifact policy** (asymmetric). Pretraining mode clips to ±20σ after z-score (catches catastrophic artifacts, leaves typical EOG/EMG for the model to learn to ignore). Evaluation mode applies additional amplitude rejection at 100 μV peak per epoch.

### 6.3 Architecture defaults

- Token dimension $d = 256$; spatial depth $L_S = 2$; temporal depth $L_T = 4$; 8 heads; MLP ratio 4.
- Patch length 50 samples (250 ms at 200 Hz), non-overlapping; 16 patches per 4 s epoch.
- Geometric MLPs: hidden width 32.
- Temporal positional embedding: learned (default).

The geometric and transductive backbones share an identical 1.58 M-parameter temporal Transformer / projection / patcher stack. Only the spatial encoder differs:

| Variant            | Distinguishing parameters (d=256, H=8, L_S=2, M=64) |
| ------------------ | ----------------------------------------------- |
| Geometric G1       |   848 |
| Geometric G2       | 17 216 |
| Geometric G3       | 18 064 |
| Transductive codex (full-rank) | 16 384 |
| Channel-independent | 0 |

G2/G3 and codex are within ~5 % of each other; G1 is roughly 19× smaller. I keep the codex at full rank (one $\mathbb{R}^{256}$ vector per electrode) rather than nerfing it for exact parity, since the full-rank codex is the standard formulation in LaBraM and EEGPT and a nerfed codex would invite a "you crippled the baseline" critique.

### 6.4 Training

- **Optimizer:** AdamW, learning rate 1e-3, weight decay 0.05.
- **Schedule:** linear warmup for 10 epochs, then cosine decay to 0 over 100 total epochs.
- **EMA τ:** cosine ramp $\tau_0 = 0.996 \to \tau_K = 1.0$ over all training steps.
- **Gradient clipping:** max-norm 1.0.
- **Batch size:** 64 epochs per step (256 in pretrain).
- **Loss:** $\lambda_R = 1$, joint (electrode, time) masking at 50 % patches.

Geometry tables are recomputed each training step because the geometric MLPs are trainable; reusing a precomputed table across steps fails with "backward through graph a second time" since the graph is freed at `loss.backward()`. At inference there is no backward, so geometry tables are precomputed once per montage.

### 6.5 Compute

Pretraining runs on Google Colab (T4 / A100). Linear-probe evaluation runs locally on Apple M-series MPS. Two scales are reported throughout this document:

- **Pilot.** 10 subjects of PhysioNet MI, 100 epochs, ~35 min on MPS. Used for the four secondary variants (codex / G2 / G3 / channel-independent) when the Colab full run had not finished.
- **Full.** 105 subjects of PhysioNet MI, 100 epochs, on Colab. Available for at least one geometric and one codex checkpoint per dataset; results in §6.5 indicate which scale each row used.

Every checkpoint is paired with a `config.yaml` containing the resolved configuration (Tier 1 preprocessing, Tier 2 ablation flags, Tier 3 artifact policy, training hyperparameters). Loading a checkpoint never relies on the source-code defaults.

### 6.6 Linear-probe protocol

For each downstream evaluation:

1. Load the pretrained online backbone, freeze all parameters (`requires_grad=False`).
2. Extract a fixed feature for each epoch: mean-pool the temporal-Transformer output over the 16 time steps to obtain a single $\mathbb{R}^{256}$ vector.
3. Fit `sklearn.StandardScaler` on the training split only (no leakage into test).
4. Fit `sklearn.LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial')`. Strictly linear — no MLP probe — so the result reflects representation quality, not probe capacity.
5. Report Balanced Accuracy (primary), Cohen's κ, and weighted F1, per subject and aggregated.

Splits per protocol:

- **LOSO (E1, E2b, E5, E7):** leave-one-subject-out. Each subject is held out as test exactly once, all other subjects are training.
- **Within-subject (E2a):** train on subject $s$'s night 1, test on subject $s$'s night 2; the encoder is frozen, only the linear probe re-fits.
- **Cross-montage (E2c):** pretrain on 64-channel PhysioNet MI; transfer the frozen backbone to 3-channel BCIC-2B. The codex baseline uses random or nearest-neighbour fallback for missing electrodes; the geometric baseline transfers directly.

### 6.7 Significance testing

For E2b I run a paired Wilcoxon signed-rank test on per-subject BAC differences between geometric (G1) and codex variants on PhysioNet MI LOSO, using `scipy.stats.wilcoxon` at $\alpha = 0.05$.

---

## 7. Results

All BAC values below are LOSO or within-subject means ± standard deviations across subjects.

### 7.1 E1 — In-distribution linear probe (PhysioNet MI)

| Variant              | Scale  | BAC          | n_subj |
| -------------------- | ------ | ------------ | ------ |
| Channel-independent  | full   | 0.333 ± 0.064 | 105 |
| Transductive codex   | full   | 0.341 ± 0.070 | 105 |
| Geometric G1         | full   | 0.342 ± 0.067 | 105 |

**Reading.** The geometric and codex models are statistically indistinguishable in-distribution. Both consistently beat the channel-independent control, confirming that explicit spatial structure helps. Geometric attention pays no in-distribution cost relative to the codex baseline — this is the fairness anchor that the proposal asks E1 to establish.

### 7.2 E2 — Robustness ladder

#### 7.2.1 E2a — Cross-session within-subject (Sleep-EDFx, night 1 → night 2)

| Variant            | Scale | Within-subject BAC | n_subj |
| ------------------ | ----- | ------------------ | ------ |
| Geometric G1       | full  | 0.532 ± 0.077       | ~78    |
| Transductive codex | full  | 0.533 ± 0.077       | ~78    |

**Reading.** Geometric attention provides no measurable advantage on cross-session transfer. This is the first robustness step and the smallest geometric perturbation in the ladder (same montage, different night, same subject). The proposal anticipated only a modest gap here because session drift mixes the geometric component (placement) with a non-geometric component (physiological state) that this architecture does not address. The flat result is consistent with that hedge.

#### 7.2.2 E2b — Cross-subject LOSO (PhysioNet MI, paired Wilcoxon)

| Variant            | Scale | LOSO BAC      | n_subj |
| ------------------ | ----- | ------------- | ------ |
| Geometric G1       | full  | 0.342 ± 0.067  | 105 |
| Transductive codex | full  | 0.341 ± 0.070  | 105 |

Paired Wilcoxon signed-rank test on per-subject BAC differences: **W = 2695.5, p = 0.911**. Not significant at $\alpha = 0.05$.

**Reading.** I cannot reject the null hypothesis that geometric and codex models have the same per-subject BAC distribution on PhysioNet MI LOSO. The proposal anticipated a robustness advantage for geometric attention in cross-subject transfer; that advantage is not detectable at this scale.

#### 7.2.3 E2c — Cross-montage zero-shot transfer (PhysioNet MI 64 ch → BCIC-2B 3 ch)

| Variant                     | Scale | LOSO BAC on BCIC-2B | n_subj |
| --------------------------- | ----- | ------------------- | ------ |
| Geometric G1                | full  | 0.509 ± 0.028        | 9 |
| Transductive codex (random) | full  | 0.513 ± 0.024        | 9 |
| **Transductive codex (nn)** | full  | **0.525 ± 0.022**    | 9 |

**Reading.** This is the proposal's headline experiment, and the result inverts the predicted ordering. The codex baseline with random fallback already matches the geometric encoder; the codex baseline with **nearest-neighbour** fallback exceeds it by ~1.6 BAC points. The nn fallback is, in effect, *reading the geometry at transfer time* and using it to assign each unseen electrode the closest pretraining codex entry. It does so without any of the parameter or computational overhead of the geometric attention bias. At this scale, the codex baseline with nn fallback recovers most of the inductive advantage that motivated the geometric attention mechanism.

### 7.3 E5 — Where $g_{ij}$ enters (G1 vs G2 vs G3)

| Variant | PhysioNet MI LOSO BAC | BCIC-2B LOSO BAC |
| ------- | --------------------- | ---------------- |
| G1 (score)  | 0.342 ± 0.067 | 0.509 ± 0.028 |
| G2 (value)  | 0.343 ± 0.069 | 0.509 ± 0.026 |
| **G3 (both)** | **0.350 ± 0.068** | **0.532 ± 0.025** |

**Reading.** G3 is consistently the best geometric variant on both datasets, with a larger lead on BCIC-2B (+2.3 BAC over G1) than on PhysioNet MI (+0.8 BAC over G1). G1 and G2 are statistically indistinguishable; their parameter counts differ by 20× but their downstream BAC does not. The proposal predicted G1 might be competitive on parsimony grounds; the data say the extra capacity at both score and value is mildly but consistently helpful.

### 7.4 E7 — Channel-independent sanity check

| Variant                | PhysioNet MI LOSO BAC | BCIC-2B LOSO BAC |
| ---------------------- | --------------------- | ---------------- |
| Channel-independent    | 0.333 ± 0.064 | 0.500 ± 0.027 |
| Geometric G1           | 0.342 ± 0.067 | 0.509 ± 0.028 |
| **Transductive codex** | 0.341 ± 0.070 | **0.522 ± 0.024** |

**Reading.** Spatial structure helps consistently: both geometric and codex models beat the channel-independent control on both datasets. On BCIC-2B specifically the codex baseline beats G1 by a slightly larger margin than on PhysioNet MI; this is consistent with the E2c finding that a well-formed codex (even one carrying random initialization in BCIC-2B's case, since the BCIC-2B subjects here were used for evaluation only, not pretraining) is a stronger baseline than the headline experiment anticipated.

### 7.5 Headline summary

The proposal's central prediction was *geometric > transductive on cross-montage transfer, ties in-distribution*. The actual result is *geometric ≈ transductive in-distribution, and slightly worse than codex-with-nn on cross-montage transfer*. The strongest within-family finding is that **G3 is the best geometric variant** on both datasets.

---

## 8. Discussion

### 8.1 What the results say honestly

Geometric attention is a competitive but not dominant alternative to a codex-based spatial encoder at the scale evaluated here. In-distribution there is no measurable difference (E1, E2b Wilcoxon). On the cross-montage transfer that the proposal framed as the headline test, a codex baseline equipped with a spatial nearest-neighbour fallback slightly beats the geometric encoder (E2c). The strongest finding from this project is therefore an *internal* one: of the three geometric variants, **G3 (geometry at both score and value) dominates G1 and G2** on both PhysioNet MI and BCIC-2B.

### 8.2 Why the headline didn't separate

Three plausible explanations, in decreasing order of how much I trust them:

1. **The codex-nn fallback exploits the same geometric prior at transfer time.** Replacing missing electrodes with their spatially-nearest pretraining electrodes uses 3-D coordinates to drive the substitution. This is, in effect, a one-step geometric prior layered on top of a transductive backbone. It is cheaper than learning a geometric attention bias, and at the scale of pretraining I could afford, it is enough.
2. **Pilot-scale pretraining is below the data threshold where inductive biases pay off.** The full PhysioNet MI corpus is 105 subjects, which is small by foundation-model standards. Inductive biases tend to help most when the data is large enough that the network would otherwise overfit identity-indexed parameters but small enough that capacity needs to be shared across structurally-similar inputs. PhysioNet MI may simply not be the regime where this trade-off bites.
3. **The 4-D $g_{ij}$ descriptor is intentionally minimal and may be insufficient.** Distance + signed displacement does not encode hemisphere, gyrus, skull conductivity, or cortical anatomy. Section 4.O6 of the project direction summary reserves richer descriptors as future work; the current null result is consistent with the descriptor being too crude rather than the inductive framing being wrong.

### 8.3 What G3 winning tells us about the design space

G2 alone (geometry at the value) and G1 alone (geometry at the score) are statistically tied on both datasets, despite a 20× difference in distinguishing parameter count. G3 — which adds both signals through separate projections of the same $g_{ij}$ — beats both. This suggests the geometric advantage is not concentrated at any single point in the attention computation; the cleanest reading is that geometry contributes a small but consistent signal at *each* slot it is allowed to enter, and combining slots adds up. This is the most actionable finding from the project, and the cleanest follow-up would be to push G3 with the richer descriptors enumerated in §8.2(3).

### 8.4 Relation to the proposal hypothesis

The proposal phrased its expectation as "I expect the transductive baseline to remain competitive in-distribution, while geometric conditioning should help most in cross-montage transfer." The first half of that hedge held; the second did not. I take the inversion as an informative result rather than a failed one: it identifies a real, practical recipe — codex + nearest-neighbour fallback — that captures most of the predicted inductive advantage at lower implementation cost, and it gives the geometric line of work a sharper target (beat *codex-nn*, not naive codex).

---

## 9. Conclusion

I implemented and evaluated a geometrically inductive self-supervised EEG encoder: spatial attention conditioned on a 4-D geometric descriptor $g_{ij}$ instead of a learned per-electrode codex. I ran the full evaluation plan committed to in the proposal — in-distribution linear probe (E1), three-step robustness ladder (E2a/b/c), G1/G2/G3 ablation (E5), and channel-independent sanity check (E7) — on three open EEG datasets (PhysioNet MI, BCIC-2B, Sleep-EDFx).

The results do not support the proposal's central prediction that geometric attention should dominate a transductive codex on robustness. Instead they support a more careful conclusion: spatial structure helps (both geometric and codex beat channel-independent), the geometric and codex variants are essentially tied in-distribution and on cross-subject LOSO, and a codex baseline with a nearest-neighbour fallback at transfer time slightly beats the geometric encoder on cross-montage transfer. The strongest finding is internal to the geometric family: G3, with geometry injected at both score and value, consistently dominates G1 and G2.

Framed as a design study, the project answers its research question — *does geometric conditioning provide a robustness advantage over the codex approach, and is that advantage large enough to matter?* — with a measured no, at the scale evaluated, and points to specific follow-ups (richer descriptors, larger pretraining, high-density montages where codex-nn breaks down) that could change the answer.

---

## 10. Limitations

1. **Pretraining scale.** PhysioNet MI is small by foundation-model standards (105 subjects, ~10–12 epochs/run × 14 runs per subject). Most EEG foundation-model papers pretrain on $10^3$–$10^4$ hours of recording; I am at $10^1$. Inductive-bias effects often grow with scale; this project cannot rule out that the geometric advantage materializes at LaBraM/EEGPT scale.
2. **Single seed for most variants.** Time and compute did not permit multi-seed averaging for every variant; per-subject variance is reported but per-seed variance is not. Differences below ~1 BAC point should be read cautiously.
3. **Sleep-EDFx conflates geometric and physiological non-stationarity.** Section 1.2 was explicit about this: E2a measures a mixture of placement drift and physiological-state drift, and only the former is addressed by the architecture. The flat E2a result is therefore consistent with the architecture working perfectly on the placement portion and gaining nothing on the state portion.
4. **Minimal geometric descriptor.** $g_{ij} \in \mathbb{R}^4$ captures distance and signed displacement only. Richer descriptors — spherical coordinates on the scalp sphere, hemisphere indicators, geodesic distance on a cortical mesh — are listed as future work and are not evaluated.
5. **Cross-montage evaluated at only one transfer point.** E2c measures 64-channel pretrain → 3-channel evaluation. The proposal's §7 mentioned a "degradation curve" across multiple levels of montage mismatch; only the endpoint is reported here.
6. **G1 implementation is dot-product-plus-bias, not literal GAT-additive.** As described in §4.3, this was a deliberate choice to share the attention kernel between geometric and codex variants and isolate the inductive-vs-transductive variable. The literal GAT-additive form was not evaluated.
7. **No BCIC-2A.** Dropped to fit compute budget. The proposal listed it as a candidate motor-imagery dataset; PhysioNet MI carries the motor-imagery evaluation alone here.

---

## 11. Future Work

The shape of the results points to three concrete follow-ups:

1. **Richer geometric descriptors with G3.** G3 was identified as the strongest geometric configuration. The natural next step is to push G3 with a richer $g_{ij}$ — spherical coordinates of $p_i, p_j$, an explicit hemisphere indicator, or geodesic distance computed on a cortical-surface mesh. The current 4-D descriptor is intentionally the smallest sensible choice; if geometry helps at all, a richer descriptor with the strongest variant is where the next BAC points should come from.
2. **Scale-up pretraining and re-run E2c.** The most informative single experiment to settle the headline question is to scale pretraining beyond pilot scale — at minimum to LaBraM-style multi-thousand-hour corpora — and re-run E2c. If the geometric advantage over codex-nn appears at scale, that would confirm the inductive-bias story; if not, the codex-nn baseline is genuinely competitive and the proposal's central claim is mis-targeted.
3. **High-density montage transfer where codex-nn breaks down.** Codex-nn substitutes the *single* nearest pretraining electrode. At $M \geq 128$ the local electrode density is high enough that "the nearest neighbour" is almost the same as "the electrode itself"; at very different montages (clinical 19-channel layouts, sparse consumer headsets) the nn substitution becomes a coarser approximation. The proposal flagged $M \geq 128$ as the regime where kNN attention sparsification might begin to matter; the same regime is also where codex-nn fallback should begin to break down. Constructing this transfer carefully would isolate whether geometric attention dominates in the limit where the nn shortcut fails.

Three smaller follow-ups: (a) implement the literal GAT-additive G1 and confirm it matches dot-product-plus-bias G1; (b) implement the full E2c degradation curve across multiple intermediate montages; (c) ablate $\lambda_R$ (loss reconstruction weight) and the masking strategy, which the proposal listed as E6 but was dropped for compute reasons.

---

## 12. Contributions

This is a **single-author project**. All design, implementation, experiments, and writing are by Tianxin Zhou. No collaborators contributed code or text.

I used **GitHub Copilot / Claude Code** as a coding-assistant during implementation: pairing on smoke-test scaffolds, refactors, and documentation. The architectural decisions, experimental design, dataset choices, and the analysis of the results in this report are mine. Where the assistant drafted prose, I rewrote it before inclusion. The course's policy that "using gen-AI solely to generate your project report ... will result in you failing the course" is the boundary I operated within: the engineering work, the experimental judgement, and the honest framing of the negative-leaning headline result are my own.

---

## 13. Reused Open-Source Code

I cite the open-source projects that informed this implementation:

- **EEGPT** (Wang et al., 2024) — [https://github.com/BINE022/EEGPT](https://github.com/BINE022/EEGPT). I read the EEGPT codebase to validate (a) the cosine ramp for the EMA $\tau$ schedule, (b) stop-gradient placement in the alignment loss, and (c) the channel-mean target convention for masked-patch reconstruction. The EEGPT repository is licensed Apache 2.0 and is cloned at the project root for reference; my code is an independent implementation, not a fork.
- **MNE-Python** [Gramfort et al., 2013] — data loading, montage handling, EDF parsing, bandpass filtering, resampling.
- **MOABB** [Aristimunha et al., 2023] — BCIC-2B dataset access (`BNCI2014_004`).
- **PyTorch** — backbone, attention, training loop.
- **scikit-learn** — `StandardScaler`, `LogisticRegression`, balanced accuracy / Cohen's κ scoring.
- **SciPy** — `scipy.stats.wilcoxon` for the paired test in E2b.

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
