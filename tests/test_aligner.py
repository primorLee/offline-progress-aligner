import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from offline_progress_aligner import FilterConfig, OfflineProgressAligner, TokenSequence
from offline_progress_aligner.aligner import filter_correspondences
from offline_progress_aligner.data import pack_sequences
from offline_progress_aligner.dtw import alignment, hard_dtw
from offline_progress_aligner.model import RelationTemporalAdapter
from offline_progress_aligner.objective import objective
from offline_progress_aligner.train import LOSS_CONFIG, load_manifest
from offline_progress_aligner.vendor.host_smooth_dtw import smooth_dtw_probs

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights/aligner_step008308.safetensors"
META = json.loads((ROOT / "weights/model.json").read_text(encoding="utf-8"))
SPACE = META["feature_encoder_checkpoint_sha256_observed"][0]
torch.set_num_threads(2)


def sequence(n=16, pairs=3, space=SPACE, seed=17):
    return TokenSequence(np.random.default_rng(seed).normal(size=(n*pairs, 512)).astype(np.float32),
                         np.arange(n+1)*pairs, np.linspace(2, 18, n), np.arange(n)*5, space)


@pytest.fixture(scope="module")
def aligner():
    return OfflineProgressAligner.from_checkpoint(WEIGHTS)


def test_published_weight_checksum_and_size(aligner):
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == META["sha256"]
    assert sum(p.numel() for p in aligner.model.parameters()) == 182657
    assert all(not p.requires_grad for p in aligner.model.parameters())


def test_tampered_weights_fail(tmp_path):
    file = tmp_path / WEIGHTS.name
    file.write_bytes(WEIGHTS.read_bytes()[:-8])
    with pytest.raises(ValueError, match="SHA"):
        OfflineProgressAligner.from_checkpoint(file, metadata=ROOT / "weights/model.json")


def test_npz_roundtrip_ragged_and_clocks(tmp_path):
    s = sequence()
    s.offsets = np.r_[0, 2, 6, np.arange(3, 17)*3]
    p = tmp_path / "input.npz"
    s.save(p)
    t = TokenSequence.load(p)
    np.testing.assert_array_equal(s.features, t.features)
    np.testing.assert_array_equal(s.offsets, t.offsets)
    np.testing.assert_array_equal(s.timestamps, t.timestamps)
    assert s.feature_space == t.feature_space


@pytest.mark.parametrize("offsets", [np.array([0, 0, 3]), np.array([1, 3]), np.array([0, 4]), np.array([0., 3.])])
def test_invalid_offsets_rejected(offsets):
    with pytest.raises(ValueError):
        TokenSequence(np.zeros((3, 512), np.float32), offsets)


def test_nonfinite_and_clock_checks():
    x = np.ones((4, 512), np.float32); x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TokenSequence(x, np.arange(5))
    with pytest.raises(ValueError, match="timestamps"):
        TokenSequence(np.ones((4, 512), np.float32), np.arange(5), np.array([3, 2, 1, 0], dtype=np.uint32))


def test_feature_space_gate(aligner):
    with pytest.raises(ValueError, match="Feature space"):
        aligner.align(sequence(space="arbitrary_encoder"), sequence())
    r = aligner.align(sequence(space="arbitrary_encoder"), sequence(), allow_feature_mismatch=True)
    assert not r["feature_space_verified"]


def test_permutation_padding_and_causal_context(aligner):
    s = sequence()
    x, valid = pack_sequences([s])
    with torch.inference_mode():
        expected = aligner.model(x, valid)
        torch.testing.assert_close(aligner.model(x.flip(2), valid), expected, atol=2e-6, rtol=2e-6)
        padded = torch.cat([x, torch.full_like(x[:, :, :1], 5.)], dim=2)
        mv = torch.cat([valid, torch.zeros_like(valid[:, :, :1])], dim=2)
        torch.testing.assert_close(aligner.model(padded, mv), expected, atol=2e-6, rtol=2e-6)
        changed = x.clone(); changed[:, 8:] *= -3
        torch.testing.assert_close(aligner.model(changed, valid)[:, :8], expected[:, :8], atol=2e-6, rtol=2e-6)


def test_known_warp_dtw_and_singletons():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(48, 128)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    warp = np.rint(np.linspace(0, 1, 40)**1.3 * 47).astype(int)
    mapped, _, path, _ = alignment(x[warp], x)
    assert np.abs(mapped-warp).mean() < 1
    assert path[0] == (0, 0) and path[-1] == (39, 47)
    assert np.all(np.diff(mapped) >= 0)
    assert hard_dtw(np.zeros((1, 1)))[0].tolist() == [0.]


def test_ambiguous_and_no_distant_candidates_rejected():
    for n in (1, 4, 16):
        f = filter_correspondences(np.arange(n), np.arange(n), np.ones((n, n)), FilterConfig())
        assert not any(f["accepted"])
        json.dumps(f, allow_nan=False)


def test_real_clock_units_and_json(aligner):
    human = sequence()
    robot = sequence()
    robot.timestamps = np.linspace(100, 108, len(robot))
    result = aligner.align(human, robot)
    np.testing.assert_allclose(result["matched_human_seconds"], human.timestamps)
    assert result["robot_seconds"][0] == 100
    assert result["human_verified"] is False
    assert result["accepted_count"] > 0
    json.dumps(result, allow_nan=False)


def test_training_objective_has_finite_gradients():
    # Random initialized CPU unit test; does not optimize or modify the released model.
    torch.manual_seed(9)
    model = RelationTemporalAdapter()
    x = torch.randn(2, 12, 3, 512)
    valid = torch.ones(x.shape[:-1], dtype=torch.bool)
    a = model(x, valid); b = model(x + .05*torch.randn_like(x), valid)
    positions = torch.linspace(0, 1, 12)[None].expand(2, -1)
    loss, _ = objective(a, b, positions, positions, LOSS_CONFIG, smooth_dtw_probs)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_leakage_prevented(tmp_path):
    sequence().save(tmp_path / "a.npz")
    rows = [dict(pair_id=str(i), task_id="cup", group_id=str(i), split=split, human="a.npz", robot="a.npz")
            for i, split in enumerate(("train", "validation"))]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="training and validation"):
        load_manifest(path)


def test_cli_end_to_end(tmp_path):
    sequence().save(tmp_path / "human.npz")
    sequence().save(tmp_path / "robot.npz")
    output = tmp_path / "result.json"
    run = subprocess.run([sys.executable, "-m", "offline_progress_aligner", "align",
        "--human", str(tmp_path / "human.npz"), "--robot", str(tmp_path / "robot.npz"),
        "--weights", str(WEIGHTS), "--output", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert output.with_suffix(".csv").is_file()
    assert json.loads(output.read_text())["accepted_count"] > 0
