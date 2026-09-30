"""Frozen adapter + two-way DTW + transparent correspondence filtering."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .data import TokenSequence, pack_sequences
from .dtw import alignment
from .model import RelationTemporalAdapter


@dataclass(frozen=True)
class FilterConfig:
    max_cycle_sampled_frames: float = 2.0
    min_cosine: float = 0.5
    min_distant_margin: float = 0.01
    excluded_human_neighbor_radius: float = 3.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Filter thresholds must be finite")
        if self.max_cycle_sampled_frames < 0 or self.excluded_human_neighbor_radius < 0:
            raise ValueError("Cycle tolerance and exclusion radius cannot be negative")
        if not -1 <= self.min_cosine <= 1 or not 0 <= self.min_distant_margin <= 2:
            raise ValueError("Cosine must be in [-1,1], margin in [0,2]")


def filter_correspondences(mapped, reverse, similarity, policy):
    n, m = similarity.shape
    matched = np.rint(mapped).astype(int).clip(0, m - 1)
    cycle = np.abs(np.interp(mapped, np.arange(m), reverse) - np.arange(n))
    assigned = similarity[np.arange(n), matched]
    far = np.abs(np.arange(m)[None, :] - mapped[:, None]) > policy.excluded_human_neighbor_radius
    competitor = np.where(far, similarity, -np.inf).max(1)
    margin = assigned - competitor
    # No distant competitor -> undefined confidence -> reject, rather than accept +inf.
    accepted = ((cycle <= policy.max_cycle_sampled_frames)
                & (assigned >= policy.min_cosine)
                & (margin >= policy.min_distant_margin) & np.isfinite(margin))
    return dict(
        nearest_human_sample_indices=matched.tolist(),
        matched_cosine=assigned.tolist(),
        cycle_return_sampled_frames=cycle.tolist(),
        distant_candidate_margin=[float(v) if np.isfinite(v) else None for v in margin],
        accepted=accepted.tolist(),
        accepted_robot_sample_indices=np.flatnonzero(accepted).tolist(),
        accepted_count=int(accepted.sum()),
    )


class OfflineProgressAligner:
    def __init__(self, model, metadata, device="cpu", max_dtw_cells=4_000_000):
        self.model = model.to(device).eval().requires_grad_(False)
        self.metadata = metadata
        self.device = device
        if max_dtw_cells < 1:
            raise ValueError("max_dtw_cells must be positive")
        self.max_dtw_cells = max_dtw_cells

    @classmethod
    def from_checkpoint(cls, weights, *, metadata=None, device="cpu"):
        weights = Path(weights)
        meta_path = Path(metadata) if metadata else weights.with_name("model.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("architecture") != "RelationTemporalAdapter"
                or meta.get("input_dim") != 512 or meta.get("hidden_dim") != 128):
            raise ValueError("Unsupported checkpoint architecture")
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != meta.get("sha256"):
            raise ValueError("Weight SHA-256 does not match model.json")
        model = RelationTemporalAdapter(meta["hidden_dim"])
        state = load_file(str(weights), device="cpu")
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Weights contain nonfinite tensors")
        model.load_state_dict(state, strict=True)
        return cls(model, meta, device)

    @torch.inference_mode()
    def embed(self, sequence: TokenSequence):
        x, valid = pack_sequences([sequence], self.device)
        value = self.model(x, valid)[0].float().cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite embeddings; check feature values and scale")
        return value

    def align(self, human: TokenSequence, robot: TokenSequence, *,
              policy=None, allow_feature_mismatch=False):
        policy = policy or FilterConfig()
        if len(human) * len(robot) > self.max_dtw_cells:
            raise ValueError("Too many DTW cells; subsample both sequences before aligning")
        expected = self.metadata.get("feature_encoder_checkpoint_sha256_observed", [])
        verified = human.feature_space == robot.feature_space and human.feature_space in expected
        if not verified and not allow_feature_mismatch:
            raise ValueError("Feature space does not match this checkpoint. Use the matching "
                             "encoder, train your own adapter, or explicitly allow a diagnostic mismatch.")
        h, r = self.embed(human), self.embed(robot)
        mapped, sim, path, cost = alignment(r, h)
        reverse, _, _, _ = alignment(h, r)
        filtered = filter_correspondences(mapped, reverse, sim, policy)
        human_frames = human.frame_indices if human.frame_indices is not None else np.arange(len(human))
        robot_frames = robot.frame_indices if robot.frame_indices is not None else np.arange(len(robot))
        result = dict(
            schema="offline_progress_correspondence_v1",
            direction="robot_to_human", method="temporal adapter + endpoint-constrained monotonic DTW",
            checkpoint_step=self.metadata.get("step"), checkpoint_sha256=self.metadata["sha256"],
            human_feature_space=human.feature_space, robot_feature_space=robot.feature_space,
            feature_space_verified=verified,
            pseudo_labels=True, human_verified=False,
            same_task_pair_assumed=True, calibrated_confidence=False,
            human_sample_count=len(human), robot_sample_count=len(robot),
            human_source_frame_indices=human_frames.tolist(),
            robot_source_frame_indices=robot_frames.tolist(),
            source_frame_ids_provided=dict(human=human.frame_indices is not None,
                                           robot=robot.frame_indices is not None),
            matched_human_sample_indices=mapped.tolist(),
            matched_human_source_frames=np.interp(mapped, np.arange(len(human)), human_frames).tolist(),
            path=[[int(i), int(j)] for i, j in path], path_cost=cost,
            confidence_policy=asdict(policy), **filtered,
        )
        if human.timestamps is not None:
            result["human_sampled_seconds"] = human.timestamps.tolist()
            result["matched_human_seconds"] = np.interp(mapped, np.arange(len(human)), human.timestamps).tolist()
        if robot.timestamps is not None:
            result["robot_seconds"] = robot.timestamps.tolist()
        return result
