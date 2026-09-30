# Training the temporal adapter

The released **training objective is the original objective**. The compact trainer below is a new portable entry point over static feature files. It does not reproduce the old cluster's asynchronous cache preparation, growing dataset or staged optimizer history bit-for-bit. The released weight is exported from that historical run; it was not retrained for this repository.

## Data manifest

Provide a JSONL manifest, with paths relative to the manifest directory:

```json
{"pair_id":"train_a","task_id":"place_cup","group_id":"episode_a","split":"train","human":"train_a/human.npz","robot":"train_a/robot.npz"}
{"pair_id":"val_b","task_id":"place_cup","group_id":"episode_b","split":"validation","human":"val_b/human.npz","robot":"val_b/robot.npz"}
```

Both sides of each row must perform the same task in the same stage order. The trainer does not infer this from `task_id`; it is a data-curation requirement. Keep all views and derivatives of one source episode in the same `group_id` and split. Group/file/content overlap between train and validation is rejected. Test data is not a supported training-manifest split. Video near-duplicate checking remains a responsibility of dataset preparation.

Use one frozen encoder/version for all input features, and at least 32 valid sampled frames per video. Train on human and robot sequences with their own time/frame axes. Positions are normalized independently within each sequence, used in the cycle target only, and never fed into the adapter.

## Objective

For each direction, a SmoothDTW soft correspondence projects source embeddings onto the other sequence. A soft nearest-neighbor return distribution maps the projected embeddings back to the source. If its mean is `mu`, variance is `var` (clamped to 1e-4), and the known originating normalized position is `p`, the cycle term is:

```
mean((mu - p)^2 / var + 0.001 * log(var))
```

The two directions are averaged and `0.3 × path_loss` is added. Similarity temperature is 0.1; SmoothDTW parameters are `gamma_s=1.0`, `gamma_f=0.1`, bidirectional, anti-diagonal implementation. `path_loss` is normalized by the two sequence lengths. A low cycle error alone does not establish semantic alignment.

## Reference training command

Run only when you intend to train; installation and inference do not start training:

```bash
python -m offline_progress_aligner.train --manifest data/pairs.jsonl --output runs/my_adapter --steps 2000 --batch-size 32 --sample-frames 32 --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 100 --device cuda:0
```

CPU is the default device if `--device` is omitted. AdamW uses weight decay 1e-4 and clips gradient norm at 3. Each epoch traverses every manifest training pair before reshuffling. Both sequences are independently time-resampled. Validation covers all manifest validation rows at each save interval; it uses fixed evenly sampled frames. Encoder features remain frozen. This trainer does not start or manage distributed workers.

The output directory must be empty. `--initial-weights` accepts this package's safetensors format and matching `model.json` for **model-only warm start**; it creates a new optimizer and LR schedule. It is not an exact continuation of the original step-8308 optimizer. The reference trainer saves `training_state.pt` for archival but does not currently provide an exact-resume CLI. Generated full states stay local and are not needed by inference.

Each saved subdirectory includes portable `model.safetensors`, its checksum/metadata and full local training state. `metrics.jsonl` records actual train/validation values. The released historical recipe and coverage are documented in [MODEL_CARD.md](../MODEL_CARD.md).
