import argparse
import csv
import json
from pathlib import Path

import torch

from .aligner import FilterConfig, OfflineProgressAligner
from .data import TokenSequence


def write_result(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    names = ["robot_sample_index", "robot_source_frame", "matched_human_source_frame",
             "robot_seconds", "matched_human_seconds", "cosine", "cycle_return_sampled_frames",
             "distant_candidate_margin", "accepted"]
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for i in range(result["robot_sample_count"]):
            writer.writerow([
                i, result["robot_source_frame_indices"][i], result["matched_human_source_frames"][i],
                result.get("robot_seconds", [None] * result["robot_sample_count"])[i],
                result.get("matched_human_seconds", [None] * result["robot_sample_count"])[i],
                result["matched_cosine"][i], result["cycle_return_sampled_frames"][i],
                result["distant_candidate_margin"][i], result["accepted"][i],
            ])


def main():
    p = argparse.ArgumentParser(description="Match and filter same-progress frames from cached 512-D tokens.")
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("align", help="Align a same-task human/robot pair (NPZ inputs)")
    for name in ("human", "robot", "weights", "output"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--metadata", help="Defaults to model.json next to weights")
    a.add_argument("--device", default="cpu")
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--allow-feature-mismatch", action="store_true")
    a.add_argument("--max-cycle-frames", type=float, default=2)
    a.add_argument("--min-cosine", type=float, default=.5)
    a.add_argument("--min-margin", type=float, default=.01)
    a.add_argument("--exclude-radius", type=float, default=3)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        aligner = OfflineProgressAligner.from_checkpoint(args.weights, metadata=args.metadata, device=args.device)
        result = aligner.align(TokenSequence.load(args.human), TokenSequence.load(args.robot),
            policy=FilterConfig(args.max_cycle_frames, args.min_cosine, args.min_margin, args.exclude_radius),
            allow_feature_mismatch=args.allow_feature_mismatch)
        write_result(result, args.output)
    except (ValueError, OSError) as error:
        p.exit(2, f"Input error: {error}\n")
    print(f"Aligned {result['robot_sample_count']} robot samples; accepted {result['accepted_count']}. "
          f"Saved JSON + CSV. Heuristic acceptance is not verified accuracy.")


if __name__ == "__main__":
    main()
