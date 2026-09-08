"""Create deterministic scene-level train, validation, and test manifests."""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split CurvePT scenes into manifests")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    scenes = sorted(path.name for path in args.dataset_root.glob("scene_*") if path.is_dir())
    if not scenes:
        raise ValueError(f"No scene_* directories found in {args.dataset_root}")

    random.Random(args.seed).shuffle(scenes)
    train_end = int(len(scenes) * args.train_ratio)
    val_end = train_end + int(len(scenes) * args.val_ratio)
    splits = {
        "train": scenes[:train_end],
        "val": scenes[train_end:val_end],
        "test": scenes[val_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_scenes in splits.items():
        manifest = {
            "dataset_root": str(args.dataset_root),
            "seed": args.seed,
            "scenes": split_scenes,
        }
        output_path = args.output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"{name}: {len(split_scenes)} scenes -> {output_path}")


if __name__ == "__main__":
    main()