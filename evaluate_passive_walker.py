import argparse
import os
from os.path import abspath, dirname, join, splitext

import numpy as np

from Introduction0b import PassiveWalkerWorld


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Passive Walker morphology and generate a video."
    )
    parser.add_argument(
        "--x-best",
        required=True,
        help="Path to x_best.npy containing the best genotype.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Output video path (.mp4). Defaults to <x_best_dir>/best_eval.mp4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    x_best_path = abspath(args.x_best)
    if not os.path.isfile(x_best_path):
        raise FileNotFoundError(f"x_best file not found: {x_best_path}")

    best_individual = np.load(x_best_path)

    video_path = args.video
    if video_path is None:
        x_best_dir = dirname(x_best_path)
        video_path = join(x_best_dir, "best_eval.mp4")
    else:
        video_path = abspath(video_path)

    output_dir = dirname(video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    world = PassiveWalkerWorld()
    world.update_robot_xml(best_individual)

    if splitext(video_path)[1].lower() != ".mp4":
        video_path = f"{video_path}.mp4"

    print(f"Loaded genotype: {x_best_path}")
    print(f"Generating video: {video_path}")
    world.generate_best_individual_video(
        env=world.create_env(),
        video_name=video_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
