import os
from glob import glob
import cv2

def create_video_from_images_cv2(image_dir, start_idx=None, end_idx=None, fps=30, output_path=None):
    image_dir = os.path.abspath(image_dir)
    folder_name = os.path.basename(image_dir.rstrip('/'))

    if output_path is None:
        output_path = os.path.join(image_dir, f"{folder_name}.mp4")
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect images
    image_paths = sorted(glob(os.path.join(image_dir, '*.*')), key=lambda x: x.lower())
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    # Select frame range
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx or 0
        end_idx = end_idx or len(image_paths)
        image_paths = image_paths[start_idx:end_idx]

    # Read the first image to get size
    frame = cv2.imread(image_paths[0])
    height, width, channels = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video written to: {output_path} (FPS={fps}, Frames={len(image_paths)})")


if __name__ == '__main__':
    image_dir = "/mnt/yubo/emily/motion/output/smpl/smpl"
    output_path = "/mnt/yubo/emily/motion/output/smpl/smpl_31_1600.mp4"
    fps = 30

    # Choose the range here:
    create_video_from_images_cv2(image_dir, start_idx=31, end_idx=1600, fps=fps, output_path=output_path)