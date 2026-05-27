# v2 Experiment Protocol

Canonical specification of the v2 cross-montage experiment. The pretrain
notebook (`notebooks/colab_v2_pretrain.ipynb`) and the upcoming v2 eval
notebook both follow this protocol mechanically. If anything here changes,
update both notebooks.

## Headline question

**Does geometric attention let a model trained on a mix of montages
generalize zero-shot to a held-out montage?**

The answer is a 3-row × 3-column table of zero-shot BAC:

| Held-out (eval) | `v2_geometric_<split>` | `v2_chind_<split>` | `v2_codex_<split>` |
|---|---|---|---|
| Sleep-EDFx     | (probe BAC) | (probe BAC) | (codex+fallback BAC) |
| BCIC-2B        | (probe BAC) | (probe BAC) | (codex+fallback BAC) |
| PhysioNet MI   | (probe BAC) | (probe BAC) | (codex+fallback BAC) |

If `v2_geometric` beats `v2_chind` on the majority of rows, the
geometric attention story is supported. Codex is the additional
"geometry is necessary for cross-montage" demonstration.

## Fair-comparison constraints

Every variant in the table is trained under **identical** conditions:

| Knob | Value |
|---|---|
| Subjects per dataset | **9** (subject-level balancing; Option A) |
| Epochs per dataset per pass | **4000** (epoch-budget cap) |
| Pretrain dataset pair | the two non-held-out datasets |
| Architecture | same `arch` block in `configs/v2/pretrain/v2_default.yaml` |
| Optimizer / LR / schedule | same `train` block |
| Mask ratio / scheme | same `ablation` block (`temporal`, 0.5) |
| Loss | `L_align + λ_R · L_recon`, per-channel target |
| EMA tau schedule | cosine, base 0.996, final 1.0 |
| Total passes | 100 epochs |

Only the **spatial encoder** differs:

- `v2_geometric` — `GeometricSpatialEncoderV2(geometry_injection='g3', g_dim=10)`
- `v2_chind` — `GeometricSpatialEncoderV2(geometry_injection='none')`
- `v2_codex` — `TransductiveSpatialEncoderV2` (per-montage codex; **not yet implemented**)

## Subject design rationale (Option A)

Each pretrain run uses 9 subjects from each of the two non-held-out
datasets. BCIC-2B has only 9 subjects total — that's the hard ceiling that
sets the per-dataset budget. PhysioNet has 105 and Sleep-EDFx has ~78
usable; we cap them at 9 to match BCIC, so every dataset contributes the
same amount of subject-level variability.

Why subject count and not just epoch count? Because cross-subject
generalization is the hard part of EEG, not within-subject reproducibility.
Subject capping equalizes the *diversity* each dataset contributes;
epoch capping equalizes the *gradient weight*. We do both.

The total pretrain corpus per run is 18 subjects (9 from each non-held-out
dataset). This is small by EEGPT standards but the per-channel target +
masked-patch reconstruction objectives are reasonably sample-efficient.

## Subject lists (locked)

Per dataset, the first 9 subjects from each loader's `ALL_SUBJECTS` /
`_all_valid_subjects()` list. Stable, reproducible across runs:

- `physionet_mi`: subjects 1, 2, 3, 4, 5, 6, 7, 8, 9 (none excluded)
- `bcic_2b`: subjects 1–9 (all)
- `sleep_edfx`: subjects 0, 1, 2, 3, 4, 5, 6, 7, 8

If you ever want to vary the subject set (e.g. seed sensitivity), do it
as a separate experiment and keep the canonical numbers in the table
above as the headline.

## Leave-one-out splits

Three runs per variant:

| Tag | Pretrain | Held out for eval |
|---|---|---|
| `no_sleep` | PhysioNet MI + BCIC-2B | Sleep-EDFx |
| `no_bcic`  | PhysioNet MI + Sleep-EDFx | BCIC-2B |
| `no_phys`  | BCIC-2B + Sleep-EDFx | PhysioNet MI |

For 2 implemented variants × 3 splits = **6 pretrain runs**. Adding
codex makes 9.

## Checkpoint locations

```
runs/pretrain/
  v2_geometric_no_sleep/  epoch_*.pt   v2_run.txt
  v2_geometric_no_bcic/   epoch_*.pt   v2_run.txt
  v2_geometric_no_phys/   epoch_*.pt   v2_run.txt
  v2_chind_no_sleep/      epoch_*.pt   v2_run.txt
  v2_chind_no_bcic/       epoch_*.pt   v2_run.txt
  v2_chind_no_phys/       epoch_*.pt   v2_run.txt
  v2_codex_no_*/          (deferred)
```

Each `v2_run.txt` records: variant, pretrain datasets, held_out,
epochs_per_dataset, n_subjects_per_dataset. The eval notebook reads this
to know which dataset to probe on.

## Eval protocol (to be wired into the v2 eval notebook)

For each `(variant, split)` checkpoint:

1. Load the held-out dataset using **all** its subjects (not just 9 —
   probe evaluation should use the full population for stable BAC).
2. Compute v2's fixed-scale `ch_pos` from the held-out dataset's
   `ch_names` (no per-montage normalization — this is critical for
   cross-montage consistency with what the encoder saw).
3. Freeze the encoder; train a linear probe per LOSO fold.
4. Report mean ± std BAC across folds.
5. Per-dataset eval protocol:
   - PhysioNet MI: LOSO (105 subjects)
   - BCIC-2B: LOSO (9 subjects)
   - Sleep-EDFx: within-subject night split (train on night 1, test on
     night 2; aggregate over subjects with both nights)

The probe code is `scripts/probe.py` — extend with a `--v2-checkpoint`
flag (or wrap it) so it loads `BackboneV2` and recomputes `ch_pos` with
the v2 helper.

## Codex variant: deferred design questions

Before implementing, settle:

1. **Codex storage during mixed training.** Per-montage codex (three
   separate `nn.Embedding(M_d, d)` tables, indexed by dataset name) vs.
   unified codex (one table with `sum(M_d)` slots).
   - Per-montage: easier to reason about, matches v1's transductive
     baseline. Each montage's codex only ever sees its own data.
   - Unified: shares embedding space across montages; arguably less
     consistent with the codex idea (different physical electrodes
     would compete for the same "embedding meaning").
   - **Likely choice**: per-montage. Matches v1, easier to retrofit
     fallback.

2. **Held-out fallback.** Codex has no embedding for the held-out
   dataset's electrodes. Two options:
   - `random`: sample `e_new ~ N(0, 0.02²)` per electrode. Effectively
     a no-op spatial conditioning. v1's E2c used this (got 0.522 vs
     geometric 0.509 on BCIC zero-shot — codex random happened to beat
     geometric because geometric was OOD; see v2_scope.md).
   - `nn`: for each held-out electrode, copy the embedding of the
     spatially nearest training electrode. Soft transfer.
   - **Likely choice**: report both — they bracket the codex's
     reasonable behaviour.

3. **Probe-time fine-tuning of new codex slots.** Frozen vs. allow
   gradient through them while the probe trains.
   - Frozen: pure zero-shot; matches the framing.
   - Fine-tuned: gives codex a fighting chance, but blurs "zero-shot"
     because the encoder is no longer frozen.
   - **Likely choice**: frozen. The headline is "the encoder
     generalizes," not "the encoder generalizes after a few gradient
     steps."

Once these are decided, write `src/v2/model/transductive_baseline_v2.py`
and add the three codex pretrain cells to section 7.codex of the
pretrain notebook.

## v2 completion checklist

State of things, so any fresh session can pick up. Tick items as they're
finished.

**Done.**
- [x] `src/v2/{preprocess,losses}.py` + `src/v2/model/{geometric_attention_v2,backbone_v2}.py` + `src/v2/datasets/mixed_corpus.py` + `tests/smoke_test_v2.py` (all smoke-tested).
- [x] `scripts/v2_pretrain.py` with `--variant {geometric,chind}` and `--n-subjects-per-dataset` flags.
- [x] `configs/v2/pretrain/v2_default.yaml`.
- [x] `notebooks/colab_v2_pretrain.ipynb` — 6 pretrain cells (3 splits × 2 variants) + codex placeholder.
- [x] `docs/v2/v2_scope.md` and this protocol document.

**In progress (Colab GPU sessions).**
- [ ] 6 pretrain runs producing `runs/pretrain/v2_{geometric,chind}_{no_sleep,no_bcic,no_phys}/epoch_*.pt` checkpoints.

**Not started — required to complete v2 (excluding codex).**
- [ ] **Probe v2 wrapper.** `scripts/probe.py` is hardcoded to v1's
      `Backbone`. Pick one:
      (a) extend `probe.py` with a `--v2-checkpoint` flag that loads
          `BackboneV2` and recomputes `ch_pos` via `src/v2/preprocess`; or
      (b) write `scripts/v2_probe.py` mirroring v1's structure but
          importing v2 modules. Choice (b) keeps v1 untouched, matches
          the v2-isolation discipline of `src/v2/`.
- [ ] **Per-dataset eval helpers.** For each held-out dataset, the eval
      protocol differs slightly (LOSO vs night-split). Mirror v1's
      `scripts/_eval_utils.py` pattern but for v2; or reuse it as-is if
      the v2 probe wrapper produces the same JSON schema.
- [ ] **`notebooks/colab_v2_experiment.ipynb`.** Drive-mount + repo
      clone (mirror pretrain notebook sections 1-4), then per
      `(variant, split)` checkpoint: run the v2 probe on the held-out
      dataset, save JSON to `runs/eval/v2/<split>/<variant>_<dataset>.json`.
      Final cell: aggregate to the 3-row × 3-column BAC table (codex
      column empty until variant lands).
- [ ] **Results aggregation.** `scripts/v2_results.py` reads the 6
      (or 9) eval JSONs and emits the headline table as
      `results/v2/headline.txt` + a markdown version for the report.
- [ ] **Report + slides update.** Add the v2 section to
      `final_report/report.md` and `final_report/slides.md` with the
      3×3 table and a one-paragraph interpretation.
- [ ] **Session log.** Append a v2 entry to `docs/session_log.md`
      describing what shipped.

**Deferred (codex variant — separate workstream).**
- [ ] Resolve the three design questions in §"Codex variant: deferred
      design questions" below.
- [ ] Write `src/v2/model/transductive_baseline_v2.py` and add
      `--variant codex` to `scripts/v2_pretrain.py`.
- [ ] Add 3 codex pretrain cells to section 7.codex of the notebook.
- [ ] Implement chosen fallback(s) in the v2 probe wrapper.
- [ ] Add the codex column to the headline table.

## Where to pick up if a fresh session starts

1. Read `CLAUDE.md`, then `docs/v2/v2_scope.md`, then this file.
2. Check this checklist for what's done vs not.
3. Check `runs/pretrain/` for which checkpoints exist (`ls runs/pretrain/v2_*`).
4. The smallest meaningful next step is usually the **probe v2 wrapper**
   — without it, no checkpoint can be evaluated.

## What's explicitly NOT in this protocol

- **v1 baselines as competitors.** v2's competitors are the v2
  variants themselves, trained under identical conditions. v1 numbers
  (105-subject PhysioNet pretraining) are a different experimental
  regime and should not be in the headline table. If we want a v1-style
  comparison later, it requires retraining v1 at matched 9-subject scale,
  which is a separate experiment.
- **Augmentation (synthetic head sizes / cap rotations).** Explicitly
  out of scope; see v2_scope.md.
- **Multi-seed runs.** Single seed for the headline; multi-seed only if
  the headline is borderline.
