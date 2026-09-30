"""Create artificial tokens with a known time warp. This is a software smoke test."""
from pathlib import Path
import argparse
import numpy as np
from offline_progress_aligner import TokenSequence

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", default="demo_inputs")
args = p.parse_args()
root = Path(args.output)
rng = np.random.default_rng(7)
human = rng.normal(size=(48, 3, 512)).astype(np.float32)
warp = np.rint(np.linspace(0, 1, 40) ** 1.35 * 47).astype(int)
robot = human[warp].copy()
TokenSequence(human.reshape(-1, 512), np.arange(49) * 3,
              np.linspace(0, 16, 48), feature_space="synthetic_smoke_test").save(root / "human.npz")
TokenSequence(robot.reshape(-1, 512), np.arange(41) * 3,
              np.linspace(0, 20, 40), feature_space="synthetic_smoke_test").save(root / "robot.npz")
np.save(root / "known_warp.npy", warp)
print("Saved synthetic tokens. These are not H2R/RH20T data or real evaluation results.")
