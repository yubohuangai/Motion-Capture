import os
import json
import torch
import numpy as np
from tqdm import tqdm
from easymocap.smplmodel import load_model

# === PATHS ===
input_dir = "/mnt/yubo/emily/highknees/output/smpl/smpl"
output_dir = os.path.join(input_dir, "obj")
os.makedirs(output_dir, exist_ok=True)

# === Load SMPL model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model(gender='neutral', model_type='smpl').to(device)

# === Process frames ===
json_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
print(f"Found {len(json_files)} frames.")

for fname in tqdm(json_files):
    path = os.path.join(input_dir, fname)
    with open(path, "r") as f:
        data = json.load(f)[0]

    # Convert all inputs to the same device
    poses = torch.tensor(data["poses"], dtype=torch.float32, device=device)
    shapes = torch.tensor(data["shapes"], dtype=torch.float32, device=device)
    Rh = torch.tensor(data["Rh"], dtype=torch.float32, device=device)
    Th = torch.tensor(data["Th"], dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(poses=poses, shapes=shapes, Rh=Rh, Th=Th, return_verts=True)

        # Some versions return dict, others return raw tensor
        if isinstance(output, dict) and "vertices" in output:
            verts_tensor = output["vertices"]
        else:
            verts_tensor = output  # output is directly the vertices tensor

        # Move to CPU and numpy
        if isinstance(verts_tensor, torch.Tensor):
            vertices = verts_tensor.detach().cpu().numpy()
        else:
            vertices = np.array(verts_tensor)

        # Handle batched shape [1, N, 3]
        if vertices.ndim == 3:
            vertices = vertices[0]

        faces = model.faces

    out_path = os.path.join(output_dir, fname.replace(".json", ".obj"))
    with open(out_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces + 1:  # OBJ is 1-indexed
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

print(f"Export complete! Saved {len(json_files)} .obj files to: {output_dir}")