# Step 8308 offline temporal adapter

## Identity

| Item | Value |
|---|---|
| Architecture | `RelationTemporalAdapter` |
| Input | Per-frame sets of 512-D directed object-pair tokens |
| Output | L2-normalized 128-D frame embedding |
| Parameters | 182,657 |
| Frozen checkpoint step | 8,308 cumulative optimizer updates |
| Public artifact | `weights/aligner_step008308.safetensors` |
| Public artifact bytes | 731,852 |
| Public SHA-256 | `a5907efc9933aeaa2dc40611445d2a86a05c9eb77a096a2ad3ac7385839aeeae` |
| Source complete-checkpoint SHA-256 | `88f5251f9b9ac68d4ee866a735a3deb273d2048fe5f17604de3f43e2fff287d1` |

The public file contains only the adapter tensors, copied from the verified frozen checkpoint without optimization or dtype conversion. Original optimizer/RNG states, cache receipts and private filesystem paths are excluded. The source digest identifies the internal training checkpoint; it is intentionally different from the exported safetensors digest. Machine-readable metadata is in [weights/model.json](weights/model.json).

## Architecture and supervision

A shared LayerNorm → Linear(512,128) → GELU transforms each object-pair token. Learned scalar scores pool the tokens within each frame. The network concatenates the pooled vector, its difference from the preceding frame and a causal 3-tap temporal convolution, then applies an MLP and L2 normalization. The first frame repeats its own preceding context. Pair order is irrelevant; temporal sample order is preserved. This short temporal context is measured in sampled frames, not a guaranteed number of seconds.

Training uses bidirectional temporal cycle consistency and a SmoothDTW path term. Cross-video frame matches are latent: normalized positions are used only to supervise a return to the originating frame in the same sequence. No object IDs, source frame indices or normalized clock values are model inputs. The loss is not a supervised semantic-frame matching error.

## Verified training lineage

1. **H&R / H2R pretraining, steps 1–4000.** The run retained the train/validation split: 2,027/228 candidate pairs, with 1,873/209 usable caches recorded at completion. The available-data gap was preserved. Batch 32; peak LR 3e-4, initial 100-step warmup, cosine to 3e-5 by step 2000, then 3e-5 through 4000.
2. **RH20T initial adaptation, steps 4001–7623.** A four-task preparation stage (task IDs 4, 6, 17, 46), preserving the pretrained optimizer. The planned 4000-update adaptation schedule began at 3e-5 and decayed toward 1e-5. This phase did not cover all RH20T tasks.
3. **Expanded RH20T stage, steps 7624–8308.** Batch 128; manifest candidates 10,622 training and 1,265 validation pairs. The saved sampler records **6,550 distinct visited training pairs** and 87,680 training-pair visits in this stage. It prioritizes newly available caches and cycles available pairs. The original schedule reached 1e-5 at step 8000 and held that floor. Training then stopped; this checkpoint was frozen for pseudo-label export.

All figures describe the saved run, not every downloaded dataset episode. Candidate count, ready-cache count and actual visited-pair count are different quantities. No claim of full-corpus coverage is made. The global step includes the H&R and RH20T stages.

## Recorded validation near the released checkpoint

The most recent fixed validation was at **step 8300**, on **20 RH20T pairs**, not at step 8308 and not the entire held-out set:

| Metric | Recorded value |
|---|---:|
| Total self-supervised loss | 1.3345919847 |
| Cycle term | 0.7559285760 |
| SmoothDTW path term | 1.9288779497 |
| Within-sequence normalized cycle-return MAE | 0.2106428146 |

`total = cycle + 0.3 × path`. The last row measures return to the source sequence, not human–robot semantic accuracy, seconds error or task success. This release has no manually verified cross-embodiment accuracy claim. The output filters use cycle tolerance 2 sampled frames, cosine 0.5 and distant margin 0.01; their acceptance fraction is also not accuracy.

## Intended use and limits

Use as an offline helper to propose and filter frame correspondences for **already paired, same-task** videos. The full robot sequence is required for DTW, including future robot observations. An online locator must be trained/evaluated separately with a causal observation boundary.

This model is specific to its 512-D interaction feature space. Detector/region errors, repeated stages, missing stages, pauses, different start/end stages, changed task order or unrelated tasks can break alignment. Endpoint constraints always force a path; confidence gates can still accept wrong correspondences. Validate a held-out set of semantic events and adjust thresholds to your sampling density before treating the output as labels.

## Data and licensing

- H&R from [Human2Robot](https://huggingface.co/datasets/dannyXSC/HumanAndRobot), referred to as H2R in internal run names; not the similarly named H2R-1M dataset.
- [RH20T](https://rh20t.github.io/), including non-commercial scenes. Weight use is non-commercial under [weights/LICENSE.md](weights/LICENSE.md).
- No dataset files, original encoder weights or training caches are distributed.
