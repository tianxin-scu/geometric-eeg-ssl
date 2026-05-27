# v2 Handoff Prompts

Copy-paste prompts to start fresh chat sessions for follow-up v2 work.
Both prompts are self-contained: they tell the new session which design
docs to read first, so it picks up state without re-deriving it.

If you ever wonder "where are those prompts again?" — they live here, on
GitHub. Open this file on the `v2-improvements` branch and copy.

---

## 1. Codex variant — design + implementation

**Recommended model:** Sonnet 4.6.

The codex discussion is bounded (three concrete design questions with
clear option spaces, then implementation that mirrors existing v1 + v2
patterns). Sonnet handles bounded design+impl well and is much cheaper
than Opus per turn. Save Opus for novel architectural exploration.

### Prompt to paste

```
I want to design and implement the codex variant for v2 cross-montage
pretraining. Context lives in docs/v2/v2_scope.md and
docs/v2/experiment_protocol.md — please read both first, especially the
§"Codex variant: deferred design questions" section, before responding.

Goal of this chat:
1. Resolve the three open design questions (codex storage layout,
   held-out fallback strategy, probe-time fine-tuning policy).
2. Implement src/v2/model/transductive_baseline_v2.py mirroring
   src/v2/model/geometric_attention_v2.py (per-electrode tokens, no CLS
   pool) but with a per-montage codex instead of a geometric MLP.
3. Extend scripts/v2_pretrain.py with --variant codex.
4. Add three pretrain cells to section 7.codex of
   notebooks/colab_v2_pretrain.ipynb.
5. Extend the v2 probe wrapper (TBD; check the completion checklist in
   docs/v2/experiment_protocol.md for status) with the chosen fallback.

Context to know:
- v1 transductive baseline lives in src/model/transductive_baseline.py
  — the v1 codex pattern is the reference.
- The geometric and chind variants are pretraining now on Colab; codex
  must be trained under identical fair-comparison conditions
  (9 subjects/dataset, 4096 epochs/dataset/pass, same
  architecture/optimizer/schedule).
- The fallback decision affects probe code, not pretrain code — codex
  pretraining itself can land before we settle fallback details.

I also want to revisit the g_ij descriptor design while we discuss
codex — the current 10-D [p_i, p_j, p_i-p_j, ||p_i-p_j||] choice was
made for v2.0; I want to talk through whether any richer descriptor
makes sense in light of what codex needs.

Start by reading the two design docs, then walk me through your
recommended answer to each codex design question with reasoning. Don't
implement until I've signed off on the design.
```

---

## 2. v2 completion (everything except codex)

**Recommended model:** Opus 4.7.

Broader scope (multiple files, integration with v1's probe.py conventions,
results-aggregation judgment calls). Opus's open-ended-task strength
matters here. Switch to Sonnet only if the work narrows to focused
implementation later.

### Prompt to paste

```
The v2 geometric and chind pretrain runs are complete (or partially
complete — check runs/pretrain/v2_*_*/ for checkpoint inventory). I
want to finish v2 end-to-end except for the codex variant (separate
workstream).

Read docs/v2/v2_scope.md and docs/v2/experiment_protocol.md first.
The completion checklist at the bottom of the protocol doc names
exactly what's left:
- Probe v2 wrapper (recommendation in checklist: option b,
  scripts/v2_probe.py)
- v2 eval notebook (colab_v2_experiment.ipynb)
- Results aggregation (scripts/v2_results.py)
- Headline 3-column × 3-row BAC table
- Report + slides v2 section
- Session log entry

Walk me through your plan to tackle these in order, flag any open
questions before starting, then proceed.
```

---

## Finding this file later

Two easy paths:

1. **GitHub web UI** —
   `https://github.com/tianxin-scu/geometric-eeg-ssl/blob/v2-improvements/docs/v2/handoff_prompts.md`.
   Bookmark it.
2. **Locally** — open `docs/v2/handoff_prompts.md` in your editor.
