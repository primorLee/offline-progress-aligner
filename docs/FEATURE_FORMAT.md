# Feature contract

Each video is one `.npz` file, loaded with `allow_pickle=False`:

| Key | Shape / type | Meaning |
|---|---|---|
| `features` | `[N,512]`, float16/32 | Concatenated object-pair tokens across frames |
| `offsets` | `[T+1]`, integer | Frame `t` uses `features[offsets[t]:offsets[t+1]]` |
| `feature_space` | Unicode scalar | Encoder/checkpoint/version identifier |
| `timestamps` | `[T]`, float64, optional | Seconds in this video's own clock |
| `frame_indices` | `[T]`, integer, optional | Source video frame IDs |

Offsets start at zero, end at N and increase strictly: every retained frame must have at least one valid pair. Arrays must be finite. Times and frame IDs are nondecreasing; preserve original times when dropping invalid frames. If source IDs are absent, outputs use sampled-frame indices and explicitly mark that source IDs were not supplied. Object IDs and time/frame indices never enter the learned network.

```python
import numpy as np
from offline_progress_aligner import TokenSequence

# frame_tokens[t] has shape [number_of_pairs_at_t, 512].
seq = TokenSequence(
    features=np.concatenate(frame_tokens).astype(np.float32),
    offsets=np.r_[0, np.cumsum([len(x) for x in frame_tokens])].astype(np.int64),
    timestamps=np.asarray(sampled_times_seconds, dtype=np.float64),
    frame_indices=np.asarray(sampled_video_frame_ids, dtype=np.int64),
    feature_space=your_encoder_checkpoint_sha256,
)
seq.save("human.npz")
```

The released adapter expects the feature space identified by the upstream checkpoint digest:

```
5d281bf0d89f2bbfd72ff5a14f9a40ce12534e790b0402e2ca970539c7bcc294
```

This identifies the original frozen **interaction-token encoder**, not a hash of the input file. The upstream encoder and detector/region assets are outside this release. Do not rename an incompatible encoder's identifier to bypass validation: DINO image embeddings, VAE latents and arbitrary 512-D vectors have different meanings. To use another encoder, train the adapter on that feature space. The optional mismatch flag supports explicit diagnostic experiments; it does not convert features.

The original RH20T cache sampled up to 48 frames independently on each side; training resampled 32 of those per side. Preserve that density for comparable confidence thresholds. `TokenSequence.take(indices)` supports explicit chronological subsampling; changing sample spacing changes the temporal adapter's effective context and the gates' physical-time tolerance. Inference refuses more than 4 million DTW cells by default. The DTW memory and time cost are O(human_frames × robot_frames).

No FPS equality or matching array lengths are assumed. A continuous mapped human index is interpolated on the human timestamps, while round-trip error is measured in sampled robot-frame units. This interpolation is an estimate, not an observed intermediate frame.
