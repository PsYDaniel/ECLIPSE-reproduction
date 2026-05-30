# Algorithmic Thinking

This document breaks the project into discrete **stages**, explains the **reasoning** behind each
one, and — for every stage — states **how we checked that it actually worked** before moving on.
It mirrors the structure of the AI-generated plan in [`../plan/ai-plan.md`](../plan/ai-plan.md);
section [§4](#4-parallel-to-the-ai-plan) maps the two onto each other.

The guiding principle of the whole project: **reproducibility is an engineering problem before it
is a research problem.** Each stage is small, has a single responsibility, and produces a
*checkable* artifact, so that when something breaks we know exactly which stage to blame.

---

## 1. Overview of the pipeline

```
Stage 0  Environment & repo setup        ┐
Stage 1  Install dependencies            │  "Can the code even run?"
Stage 2  Compile the custom CUDA op      ┘
Stage 3  Prepare the ADE20K dataset      ┐  "Does the model have data?"
Stage 4  Fetch checkpoints & inspect     ┘
Stage 5  Evaluate 100-50                  ┐
Stage 6  Evaluate 100-10                  │  "Do the numbers match the paper?"
Stage 7  Evaluate 100-5                   ┘
Stage 8  Bonus: threshold optimisation + plot   "Can we improve it?"
```

Each stage below lists its **goal**, **what it does**, and **how it is verified**.

## 2. Project stages (detailed)

### Stage 0 — Environment & repository setup
- **Goal:** get a working copy of the official ECLIPSE code and somewhere to store data.
- **What it does:** mounts Google Drive and clones `https://github.com/clovaai/ECLIPSE`.
- **Why:** Colab sessions are ephemeral; Drive persists checkpoints and prepared data between runs.

### Stage 1 — Install dependencies
- **Goal:** install the exact libraries the model needs.
- **What it does:** installs Detectron2 (from source), panopticapi, cityscapesScripts, opencv, and
  the repo's own `requirements.txt`.
- **Why:** Detectron2 must be compiled against the runtime's PyTorch/CUDA, so it is installed from
  git rather than a wheel.

### Stage 2 — Compile the custom CUDA operator
- **Goal:** build `MultiScaleDeformableAttention`, the custom op Mask2Former depends on.
- **What it does:** patches `ms_deform_attn_cuda.cu` (`.scalar_type().is_cuda()` → `.is_cuda()`),
  then runs `make.sh`.
- **Why:** newer PyTorch removed the old `scalar_type()` API path; without the patch the build fails
  and the model cannot start. **This is the project's critical fix.**

### Stage 3 — Prepare the ADE20K dataset
- **Goal:** turn raw ADE20K into the semantic / panoptic / instance layouts Detectron2 expects.
- **What it does:** downloads `ADEChallengeData2016.zip` and `annotations_instance.tar`, then runs
  `prepare_ade20k_sem_seg.py`, `prepare_ade20k_pan_seg.py`, `prepare_ade20k_ins_seg.py`.
- **Why:** the model reads a specific folder structure; the raw download is not in that form.
- **Engineering note:** the work is done on Colab's **fast local disk**, not Drive, because Drive
  I/O over ~22k images is the dominant bottleneck.

### Stage 4 — Fetch checkpoints & inspect the run script
- **Goal:** obtain the released weights and understand how to launch evaluation.
- **What it does:** downloads `ade_ps_100_50_final.pth` (and later the 100-10 / 100-5 weights) and
  prints `script/ade_ps/100_50.sh` to read its arguments.
- **Why:** reading the script reveals the `--eval-only` flag, the config file, and the
  `CONT.*` continual-learning arguments we must keep intact.

### Stages 5–7 — Evaluate each scenario (100-50, 100-10, 100-5)
- **Goal:** measure PQ for each continual setting and compare to the paper.
- **What it does:** on a clean local-disk clone, programmatically edits the scenario script to (a)
  point at the local dataset, (b) use the downloaded checkpoint, and (c) **comment out the long
  training command** so only the `--eval-only` evaluation runs; then runs it.
- **Why:** training each scenario takes many GPU-hours; the paper's claim we care about is the
  *evaluation* number, which the released weights let us verify directly.

### Stage 8 — Bonus: threshold optimisation + results plot
- **Goal:** test a hypothesis and visualise everything.
- **What it does:** lowers `OBJECT_MASK_THRESHOLD` from `0.5` to `0.35` across configs/code,
  re-evaluates, and plots official vs. reproduced vs. optimised PQ as a grouped bar chart
  (`results/reproduction_graph.png`).
- **Why:** novel-class masks tend to be lower-confidence; a softer threshold should recover correct
  masks that the default cut-off discards.

## 3. How each stage is tested

Reproduction fails silently if you don't check intermediate outputs. Each stage has an explicit
**pass condition**:

| Stage | Test / verification | Pass condition |
|:-----:|---------------------|----------------|
| 0 | `ls ECLIPSE` after clone; Drive mount message | repo folder exists; Drive mounted at `/content/drive` |
| 1 | `pip` exit codes; `import detectron2` | installs finish without error; import succeeds |
| 2 | `make.sh` build log; `import MultiScaleDeformableAttention` | `.so` built; import works (no CUDA symbol errors) |
| 3 | folder check + progress bars | `myade20k_panoptic_{train,val}/` created; `100% 20210/20210` and `100% 2000/2000` complete |
| 4 | file size of `.pth`; `cat` of the script | checkpoint ~hundreds of MB (not an HTML error page); script prints with `--eval-only` line present |
| 5–7 | Detectron2 evaluation log | a final **PQ** value prints, within ±0.1 of the paper (35.6 / 33.9 / 32.9) |
| 8 | `sed` confirmation message; chart file | "Threshold changed from 0.5 to 0.35" printed; `reproduction_graph.png` saved with 3×3 bars |

**General testing strategy.** Every stage either prints a progress bar that must reach 100%, creates
a file we can `ls`/size-check, or emits a number we compare against a known reference. We never
assume a stage worked because the cell "ran" — we look for the specific artifact it was supposed to
produce. This is why the pipeline is restartable: a failed check tells us precisely which stage to
re-run.

A common failure we explicitly guard against: a `wget` that "succeeds" but actually downloaded an
HTML error page instead of the real `.pth`. The file-size check in Stage 4 catches this — a real
checkpoint is hundreds of megabytes, an error page is a few kilobytes.

## 4. Parallel to the AI plan

The AI assistant produced a structured project plan *before* implementation; it lives in
[`../plan/ai-plan.md`](../plan/ai-plan.md). The mapping between that plan and the stages above is
deliberately one-to-one:

| AI plan phase | Stages here | Notes |
|---------------|-------------|-------|
| Phase 1 — "Make it run" | 0, 1, 2 | environment + the CUDA-compile risk flagged early |
| Phase 2 — "Feed it data" | 3, 4 | dataset prep + checkpoints |
| Phase 3 — "Verify the claim" | 5, 6, 7 | one scenario per task, easiest first |
| Phase 4 — "Extend it" | 8 | the bonus, only attempted after reproduction succeeded |

Where the plan and reality diverged is itself informative: the plan under-estimated the CUDA
compilation and Drive-I/O problems, which is why both became their own carefully-checked stages.
That divergence is discussed in [`ai-collaboration.md`](ai-collaboration.md) and reflected on in the
[takeaways](../takeaways.pdf).
