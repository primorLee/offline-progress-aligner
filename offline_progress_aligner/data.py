"""Portable, non-pickle feature format. Timestamps belong to each video's clock."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TokenSequence:
    features: np.ndarray
    offsets: np.ndarray
    timestamps: np.ndarray | None = None
    frame_indices: np.ndarray | None = None
    feature_space: str = "unspecified"

    def __post_init__(self):
        self.features = np.asarray(self.features)
        self.offsets = np.asarray(self.offsets)
        if self.features.ndim != 2 or self.features.shape[1] != 512:
            raise ValueError("features must have shape [total_object_pairs, 512]")
        if self.features.dtype.kind != "f" or not np.isfinite(self.features).all():
            raise ValueError("features must be finite floating-point values")
        if (self.offsets.ndim != 1 or len(self.offsets) < 2
                or self.offsets.dtype.kind not in "iu"):
            raise ValueError("offsets must be a 1-D integer array of length T+1")
        self.offsets = self.offsets.astype(np.int64)
        if (self.offsets[0] != 0 or self.offsets[-1] != len(self.features)
                or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError("Each frame needs at least one pair; offsets must span all features")
        if not isinstance(self.feature_space, str) or not self.feature_space:
            raise ValueError("feature_space must be a non-empty encoder/version identifier")
        for name in ("timestamps", "frame_indices"):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values)
            if (values.shape != (len(self),) or values.dtype.kind not in "iuf"
                    or not np.isfinite(values).all() or np.any(np.diff(values.astype(np.float64)) < 0)):
                raise ValueError(f"{name} must be finite, nondecreasing and length T")
            if name == "frame_indices" and (values.dtype.kind not in "iu" or np.any(values < 0)):
                raise ValueError("frame_indices must be nonnegative integers")
            setattr(self, name, values.copy())

    def __len__(self):
        return len(self.offsets) - 1

    def frame(self, index):
        return self.features[self.offsets[index]:self.offsets[index + 1]]

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or np.any(np.diff(indices) < 0):
            raise ValueError("Selected indices must be a nonempty, ordered 1-D sequence")
        if indices[0] < 0 or indices[-1] >= len(self):
            raise ValueError("Selected indices are outside the sequence")
        frames = [self.frame(int(i)) for i in indices]
        return TokenSequence(
            np.concatenate(frames), np.r_[0, np.cumsum([len(f) for f in frames])],
            self.timestamps[indices] if self.timestamps is not None else None,
            self.frame_indices[indices] if self.frame_indices is not None else indices,
            self.feature_space,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = dict(features=self.features, offsets=self.offsets,
                      feature_space=np.asarray(self.feature_space))
        if self.timestamps is not None:
            values["timestamps"] = self.timestamps
        if self.frame_indices is not None:
            values["frame_indices"] = self.frame_indices
        with path.open("wb") as stream:
            np.savez_compressed(stream, **values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as packet:
            required = {"features", "offsets", "feature_space"}
            if not required.issubset(packet.files):
                raise ValueError(f"NPZ must contain {sorted(required)}")
            space = packet["feature_space"]
            if space.ndim != 0 or space.dtype.kind != "U":
                raise ValueError("feature_space must be a Unicode scalar")
            return cls(packet["features"], packet["offsets"],
                       packet["timestamps"] if "timestamps" in packet else None,
                       packet["frame_indices"] if "frame_indices" in packet else None,
                       str(space.item()))


def pack_sequences(sequences, device="cpu"):
    """Pad object-pair count only; all selected sequences must have equal T."""
    if not sequences or len({len(s) for s in sequences}) != 1:
        raise ValueError("A batch needs equal nonzero frame counts")
    t = len(sequences[0])
    k = max(int(np.diff(s.offsets).max()) for s in sequences)
    pairs = torch.zeros(len(sequences), t, k, 512, dtype=torch.float32)
    valid = torch.zeros(len(sequences), t, k, dtype=torch.bool)
    for b, seq in enumerate(sequences):
        for i in range(t):
            frame = torch.from_numpy(seq.frame(i).astype(np.float32, copy=True))
            pairs[b, i, :len(frame)] = frame
            valid[b, i, :len(frame)] = True
    return pairs.to(device), valid.to(device)
