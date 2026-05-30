# Working with AI — and how to reproduce these results

This document explains **how an AI chat assistant was used** throughout the project and, for each
stage, **how you could prompt an assistant to reach the same result**. The point is not "the AI did
it" — it is to make the collaboration *transparent and repeatable*. Every prompt below was followed
by reading the output critically, running it, and feeding the real error messages back in.

> **Honest framing.** The AI was most useful for *environment plumbing* (install errors, CUDA
> compilation, dataset paths) and for *explaining* unfamiliar concepts. It was least reliable when it
> guessed at exact file paths or API details it couldn't see — those always had to be verified
> against the actual repository and error output.

## How to get good, reproducible answers

A few habits made the assistant's output trustworthy:

1. **Give it the real error, verbatim.** Pasting the full traceback beats describing it.
2. **Tell it the environment.** "Google Colab, Python 3.12, CUDA GPU, PyTorch from the default
   image" changes the answer completely.
3. **Ask for a check, not just a fix.** "…and give me a command to confirm it worked" turns every
   step into something verifiable.
4. **Make it explain before it edits.** Asking "why does this fail?" before "fix it" caught wrong
   guesses early.
5. **Keep the loop tight.** One change → run → paste result back. Large multi-step answers were
   harder to debug than small ones.

## Stage-by-stage prompts

### Stage 0–1 — Setup & dependencies
> *"I'm on Google Colab with a GPU. I want to run the clovaai/ECLIPSE repo. Give me the cells to
> clone it and install Detectron2, panopticapi, cityscapesScripts and the repo requirements, in the
> right order, and a command to confirm Detectron2 imports."*

Follow-up when an install failed: paste the `pip` error and ask *"what does this dependency conflict
mean and what's the minimal fix?"*

### Stage 2 — Compiling the CUDA op (the hard one)
> *"Running `make.sh` for MultiScaleDeformableAttention fails with this compiler error: \<paste full
> error\>. I'm on PyTorch \<version\>. What changed in the API and what's the smallest patch?"*

This is where the assistant identified that `.scalar_type().is_cuda()` is no longer valid and
suggested the one-line `sed` patch to `.is_cuda()`. **Verification it suggested:** re-run `make.sh`
and then `import MultiScaleDeformableAttention` in Python.

### Stage 3 — Dataset preparation
> *"ADE20K prep is extremely slow on Google Drive. The scripts are `prepare_ade20k_sem_seg.py`,
> `prepare_ade20k_pan_seg.py`, `prepare_ade20k_ins_seg.py`. How do I download to Colab's local disk
> instead, run them there, and copy only the results back to Drive?"*

This produced the `wget`/`unzip`/`rsync --exclude` strategy. **Verification:** check that the
`myade20k_panoptic_{train,val}` folders appear and the progress bars hit 100%.

### Stage 4 — Checkpoints & the run script
> *"Download `ade_ps_100_50_final.pth` from the ECLIPSE GitHub releases into `checkpoints/`, then
> show me the contents of `script/ade_ps/100_50.sh` so I can see how evaluation is launched."*

Reading the script together, the assistant pointed out the `--eval-only` line and which `CONT.*`
arguments must stay untouched. **Verification:** confirm the `.pth` size is hundreds of MB.

### Stages 5–7 — Evaluating each scenario
> *"I want to evaluate the released weights, not train. In `script/ade_ps/100_50.sh`, programmatically
> point the data root at my local dataset, use my downloaded checkpoint, and comment out the
> `python train_inc.py …` line that is **not** `--eval-only`. Then run it."*

The assistant produced the small Python snippet that rewrites the script (`text.replace(...)` and
commenting out the training line). **Verification:** a final `PQ` value prints; compare to the
paper. Re-used almost verbatim for 100-10 and 100-5 by swapping the checkpoint and script name.

### Stage 8 — Bonus threshold experiment
> *"Hypothesis: novel-class masks are lower-confidence, so the default `OBJECT_MASK_THRESHOLD: 0.5`
> discards correct masks. Give me a `sed` command to change it to 0.35 across the configs/code, then
> a matplotlib grouped bar chart comparing official, reproduced and optimised PQ for the three
> scenarios."*

**Verification:** the "Threshold changed from 0.5 to 0.35" confirmation prints, evaluation re-runs,
and the chart shows three groups of three bars.

## When the AI was wrong (and how we caught it)

- It occasionally invented file paths inside the repo. **Caught by** `ls`-ing the path before
  trusting it.
- It sometimes proposed reinstalling packages that were already fine, masking the real error.
  **Caught by** reading the *first* error in the traceback, not the last.
- Early plans under-estimated how long dataset prep and CUDA compilation would take. **Caught by**
  actually timing them and promoting both to their own checkpointed stages.

The broader lesson — that AI output is a *draft to verify*, not a final answer — is the main subject
of the [reflective takeaways](../takeaways.pdf).
