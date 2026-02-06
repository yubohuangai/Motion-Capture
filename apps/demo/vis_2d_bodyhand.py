"""
Visualize 2D body + hand keypoints and save images.
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from tqdm import tqdm
from easymocap.dataset import CONFIG, MV1PMF
from easymocap.mytools.debug_utils import log_time, log


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to dataset")
    parser.add_argument("--out", type=str, required=True, help="Output directory for vis images")
    parser.add_argument("--annot", type=str, default="annots", help="Annotation folder name")
    parser.add_argument("--sub", type=str, nargs="+", default=[], help="Cameras to process")
    parser.add_argument("--ext", type=str, default=".jpg", help="Image extension")
    parser.add_argument("--save_origin", action="store_true", help="Save original images in output")
    parser.add_argument("--kpts_type", type=str, default="bodyhand",
                        help="Keypoints type: bodyhand or total")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kpts_type not in CONFIG:
        raise ValueError(f"Unknown kpts_type: {args.kpts_type}")

    log_time("Starting EasyMocap 2D visualization...")
    log(f"Input path: {args.path}")
    log(f"Output directory: {args.out}")

    dataset = MV1PMF(
        args.path,
        annot_root=args.annot,
        cams=args.sub,
        out=args.out,
        config=CONFIG[args.kpts_type],
        kpts_type=args.kpts_type,
        undis=False,
        no_img=False,
        verbose=args.verbose,
    )
    dataset.writer.save_origin = args.save_origin

    for nf in tqdm(range(len(dataset)), desc="Visualizing 2D detections"):
        images, annots = dataset[nf]
        dataset.vis_detections(images, annots, nf, sub_vis=args.sub)

    log_time("Visualization complete!")


if __name__ == "__main__":
    main()
