"""Reference adapter training on a static manifest of frozen token pairs.

This is a portable training entry point, not the original cluster orchestration.
No video encoder, region detector, world model or online progress head is trained.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .aligner import OfflineProgressAligner
from .data import TokenSequence, pack_sequences
from .model import RelationTemporalAdapter
from .objective import objective
from .vendor.host_smooth_dtw import smooth_dtw_probs

LOSS_CONFIG = dict(similarity_temperature=.1, dtw_gamma=1., dtw_column_temperature=.1,
                   cycle_variance_weight=.001, path_loss_weight=.3)


def load_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups, files, content, ids = {}, {}, {}, set()
    for row in rows:
        required = {"pair_id", "task_id", "group_id", "split", "human", "robot"}
        if not required.issubset(row) or row["split"] not in ("train", "validation"):
            raise ValueError("Manifest rows need pair_id, task_id, group_id, split, human and robot; no test data")
        if row["pair_id"] in ids:
            raise ValueError("Duplicate pair_id")
        ids.add(row["pair_id"])
        split = row["split"]
        if groups.setdefault(row["group_id"], split) != split:
            raise ValueError("Same group appears in training and validation")
        for side in ("human", "robot"):
            file = (path.parent / row[side]).resolve()
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if files.setdefault(str(file), split) != split or content.setdefault(digest, split) != split:
                raise ValueError("Same feature file/content appears in training and validation")
            row[side] = str(file)
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise ValueError("Provide nonempty training and validation splits")
    return rows


@lru_cache(maxsize=256)
def cached_sequence(path):
    return TokenSequence.load(path)


def make_batch(rows, side, frames, device, rng=None):
    selected, positions = [], []
    for row in rows:
        seq = cached_sequence(row[side])
        if len(seq) < frames:
            raise ValueError("Each training sequence must contain at least sample_frames")
        u = np.linspace(0, 1, frames)
        if rng is not None:
            u = u ** float(rng.uniform(.65, 1.65))
            u[1:-1] += rng.uniform(-.25 / frames, .25 / frames, frames - 2)
            u = np.clip(np.sort(u), 0, 1)
        indices = np.rint(u * (len(seq) - 1)).astype(int)
        selected.append(seq.take(indices))
        clock = (seq.timestamps if seq.timestamps is not None else
                 seq.frame_indices if seq.frame_indices is not None else np.arange(len(seq)))
        span = float(clock[-1] - clock[0])
        if span <= 0:
            raise ValueError("Training sequences need a nonzero time/frame span")
        positions.append((clock[indices] - clock[0]) / span)
    x, valid = pack_sequences(selected, device)
    return x, valid, torch.tensor(np.asarray(positions), device=device, dtype=torch.float32)


def batch_loss(model, rows, args, rng=None):
    a, ma, pa = make_batch(rows, "human", args.sample_frames, args.device, rng)
    b, mb, pb = make_batch(rows, "robot", args.sample_frames, args.device, rng)
    return objective(model(a, ma), model(b, mb), pa, pb, LOSS_CONFIG, smooth_dtw_probs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True, help="Updates in this new run")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-frames", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=270927)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--initial-weights", help="Warm start adapter tensors; creates a NEW optimizer/run")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.sample_frames, args.save_every, args.threads) < 1:
        p.error("Steps, batch, frame count, save interval and thread count must be positive")
    if args.sample_frames < 2 or not 0 <= args.warmup_steps < args.steps:
        p.error("Need at least 2 frames; warmup must be >=0 and smaller than total updates")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        p.error("Require 0 < min learning rate <= peak learning rate")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty; existing experiments are never overwritten")
    rows = load_manifest(args.manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    spaces = {cached_sequence(r[s]).feature_space for r in rows for s in ("human", "robot")}
    if len(spaces) != 1 or "unspecified" in spaces:
        p.error("All pairs must use the same documented feature encoder/version")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.initial_weights:
        loaded = OfflineProgressAligner.from_checkpoint(args.initial_weights, device=args.device)
        if not spaces.issubset(set(loaded.metadata.get("feature_encoder_checkpoint_sha256_observed", []))):
            p.error("Warm start feature space differs from the checkpoint")
        model = loaded.model.requires_grad_(True)
    else:
        model = RelationTemporalAdapter().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    order, cursor, epoch = [], 0, 0
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            indices = []
            while len(indices) < args.batch_size:
                if cursor == len(order):
                    order = rng.permutation(len(train)).tolist()
                    cursor = 0
                    epoch += 1
                n = min(args.batch_size - len(indices), len(order) - cursor)
                indices.extend(order[cursor:cursor + n])
                cursor += n
            if step <= args.warmup_steps:
                lr = args.learning_rate * step / args.warmup_steps
            else:
                fraction = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
                lr = args.min_learning_rate + .5 * (args.learning_rate - args.min_learning_rate) * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = batch_loss(model, [train[i] for i in indices], args, rng)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite objective; stopping without an optimizer update")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            record = dict(phase="train", step=step, lr=lr, grad_norm=float(norm), **metrics)
            log.write(json.dumps(record) + "\n"); log.flush()
            if step % args.save_every != 0 and step != args.steps:
                continue
            model.eval()
            with torch.inference_mode():
                values = [batch_loss(model, [row], args)[1] for row in val]
            validation = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            validation.update(phase="validation", step=step, evaluated_pairs=len(val))
            log.write(json.dumps(validation) + "\n"); log.flush()
            print(json.dumps(validation), flush=True)
            target = out / f"step{step:06d}"
            target.mkdir()
            weights = target / "model.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights))
            metadata = dict(architecture="RelationTemporalAdapter", step=step, input_dim=512,
                hidden_dim=128, file=weights.name, sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                feature_encoder_checkpoint_sha256_observed=sorted(spaces), training_config=LOSS_CONFIG,
                metrics_are_not_cross_embodiment_semantic_accuracy=True)
            (target / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            # Private, local-only full state for future continuation; not loaded by public inference.
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
                order=order, cursor=cursor, epoch=epoch, args=vars(args)), target / "training_state.pt")


if __name__ == "__main__":
    main()
