import cv2
import os

video_path = "/Users/yubo/data/kickoff.mp4"
output_dir = os.path.splitext(video_path)[0]  # -> "/Users/yubo/data/kickoff"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, frame)
    frame_idx += 1

cap.release()
print(f"Done! Extracted {frame_idx} frames to {output_dir}")