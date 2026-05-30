# Project Documentation

**Project:** Reproducing ECLIPSE — Efficient Continual Learning in Panoptic Segmentation
**Author:** Daniel Chicherin, Or Blazer · **Course:** Python · **Date:** May 2026

This document is the full technical write-up of the project. The [README](../README.md) is the
short overview; this file explains the *why* and *how* in detail. If you only read one section to
understand the result, read [§7 Results](#7-results).

---

## 1. Motivation and problem statement

A deployed vision system rarely sees its full world on day one. New object categories appear over
time, and we would like the model to learn them **without retraining from scratch** and **without
forgetting** what it already knew. In machine learning this is called **continual** (or
*incremental*) learning, and its central obstacle is **catastrophic forgetting**: updating the
weights for new classes degrades performance on the old ones.

The problem is especially severe for **panoptic segmentation**, the task that unifies:

- **Semantic segmentation** — assign a class label to every pixel ("stuff": sky, road, wall…), and
- **Instance segmentation** — separate individual countable objects ("things": each person, car…).

Panoptic models output many interacting mask + class predictions per image, so forgetting shows up
as both wrong labels and missing or merged objects. The metric of record is **Panoptic Quality
(PQ)**, which combines segmentation accuracy and recognition accuracy into a single number.

## 2. The ECLIPSE method (what we are reproducing)

ECLIPSE (CVPR 2024) is built on top of **Mask2Former** with a **ResNet-50** backbone. Its core idea
is to avoid touching the base model at all:

1. **Freeze the base model.** After learning the first 100 classes, every backbone, pixel-decoder
   and transformer-decoder weight is frozen. Frozen weights cannot be overwritten, so old knowledge
   cannot be forgotten by gradient updates.
2. **Visual Prompt Tuning (VPT).** For each new group of classes, ECLIPSE learns a small set of
   **prompt embeddings** that are injected into the transformer decoder. Only these prompts (plus a
   light classification head) are trained — about **1.3% of the total parameters**.
3. **Logit manipulation.** Because old and new classes share visual structure, ECLIPSE nudges the
   class logits (`CONT.LOGIT_MANI_DELTAS`) to counter *semantic drift* and reduce false "background"
   predictions on novel classes.

The pay-off is strong robustness to forgetting with a fraction of the trainable parameters of full
fine-tuning, reaching state of the art on the ADE20K continual panoptic benchmark.

> **Scope note.** Training the base step is expensive. Following the paper authors' released
> checkpoints, this project **evaluates the official pre-trained weights** for each scenario rather
> than re-running the multi-day training. The reproduction therefore validates the *reported
> evaluation numbers*, which is the claim a reader most wants to trust.

## 3. The dataset: ADE20K

[ADE20K](http://sceneparsing.csail.mit.edu/) is a scene-parsing dataset with **150 labelled
categories** (100 used as the base set here, 50 held out as the novel/incremental classes). The
project uses:

- `ADEChallengeData2016.zip` — the RGB images and base annotations.
- `annotations_instance.tar` — the instance-level annotations needed for the "things" classes.

Detectron2 cannot read ADE20K directly, so three preparation scripts convert it into the layouts the
model expects:

| Script | Produces | Used for |
|--------|----------|----------|
| `prepare_ade20k_sem_seg.py` | semantic masks | "stuff" classes |
| `prepare_ade20k_pan_seg.py` | panoptic PNGs + JSON | full panoptic eval |
| `prepare_ade20k_ins_seg.py` | instance annotations | "things" classes |

In the notebook these run in ~30 minutes on a Colab GPU runtime, with the panoptic step being the
slowest (it rasterises ~22,000 panoptic PNGs).

## 4. Continual scenarios

The benchmark is defined by how the 50 novel classes are introduced after the 100 base classes:

| Scenario | Increment size | Number of steps | Total tasks | Difficulty |
|:--------:|:--------------:|:---------------:|:-----------:|:----------:|
| **100-50** | 50 classes at once | 1 | 2 | easiest |
| **100-10** | 10 classes / step | 5 | 6 | medium |
| **100-5**  | 5 classes / step  | 10 | 11 | hardest |

More steps mean more rounds of learning-and-freezing, and therefore more opportunities for
forgetting and drift — which is exactly why PQ drops as we move from 100-50 down to 100-5.

## 5. Environment and installation

The project targets a **Google Colab GPU runtime** (Python 3.12, CUDA). The dependency chain is the
fiddliest part of the whole reproduction:

- **Detectron2** — installed from source so it matches the runtime's PyTorch build.
- **panopticapi** & **cityscapesScripts** — provide the panoptic eval and label utilities.
- **MultiScaleDeformableAttention** — a custom CUDA operator that Mask2Former needs. It is compiled
  on the machine via `make.sh`. On current PyTorch the file
  `ms_deform_attn_cuda.cu` calls `.scalar_type().is_cuda()`, which no longer compiles; the notebook
  patches it to `.is_cuda()` with a one-line `sed` before building. **This patch is the single most
  important fix in the project** — without it the model will not run.

See [`requirements.txt`](../requirements.txt) for the full list and the exact install commands.

## 6. Reproduction pipeline

The notebook is organised as a clean, restartable pipeline. Each stage is described in detail, with
its verification criteria, in [`algorithmic-thinking.md`](algorithmic-thinking.md). In summary:

1. **Setup** — mount Drive, clone ECLIPSE, install dependencies.
2. **Compile** the CUDA op (with the `scalar_type` patch).
3. **Prepare ADE20K** — download images, run the three `prepare_*` scripts.
4. **Fetch checkpoints** and inspect the run script for each scenario.
5. **Evaluate** each scenario (`100-50`, `100-10`, `100-5`) in `--eval-only` mode, with the long
   training command commented out so only evaluation runs.
6. **Bonus optimisation** — lower the object-mask confidence threshold and re-measure.
7. **Plot** the official-vs-reproduced-vs-optimised comparison chart.

A recurring engineering theme: the project repeatedly **copies the work onto Colab's fast local
disk** (and rebuilds a clean clone) instead of running from Google Drive, because Drive I/O on the
20k+ image dataset is the dominant bottleneck.

## 7. Results

PQ over all classes, after the final task:

| Scenario | Official paper | Our reproduction | Δ vs. paper | Bonus (thr 0.35) |
|:--------:|:--------------:|:----------------:|:-----------:|:----------------:|
| 100-50   | 35.6 | 35.63 | +0.03 | 35.95 |
| 100-10   | 33.9 | 33.84 | −0.06 | 34.20 |
| 100-5    | 32.9 | 32.85 | −0.05 | 33.10 |

![Results chart](../results/reproduction_graph.png)

**Reading the result.** Every reproduced number is within ±0.1 PQ of the paper — the small residual
is expected run-to-run variation in evaluation, not a methodological gap. The ranking
`100-50 > 100-10 > 100-5` is preserved, confirming the expected "more tasks → more forgetting"
trend.

## 8. Bonus experiment — confidence threshold

Panoptic inference discards predicted masks whose confidence falls below
`OBJECT_MASK_THRESHOLD` (default **0.5**). The hypothesis: in the continual setting, novel-class
masks are systematically *less confident*, so a 0.5 cut-off throws away correct masks. Lowering the
threshold to **0.35** should recover some of them.

The notebook applies the change directly in the configs/code and re-evaluates. The result is a
small, consistent gain in every scenario (+0.25 to +0.35 PQ), which supports the hypothesis. The
trade-off — admitting more low-confidence masks risks extra false positives — is discussed in the
[takeaways](../takeaways.pdf).

## 9. Conclusions

- The ECLIPSE evaluation results **reproduce faithfully** (±0.1 PQ) across all three scenarios.
- The hardest part of reproduction was **not** the science but the **environment**: CUDA
  compilation, dataset preparation, and Colab I/O. The `scalar_type` patch is essential.
- A simple, well-motivated tweak (lower mask threshold) yields a small but reliable improvement,
  illustrating that reproduction can become genuine experimentation.

## 10. References

1. B. Kim, J. Yu, S. J. Hwang. *ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with
   Visual Prompt Tuning.* CVPR 2024. arXiv:2403.20126.
2. B. Cheng et al. *Masked-attention Mask Transformer for Universal Image Segmentation
   (Mask2Former).* CVPR 2022.
3. B. Zhou et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
4. A. Kirillov et al. *Panoptic Segmentation.* CVPR 2019 (defines the PQ metric).
5. Y. Wu et al. *Detectron2.* https://github.com/facebookresearch/detectron2
