import os
import json
import torch
import numpy as np
import trimesh
from easymocap.smplmodel import load_model

# === Path to one JSON file ===
json_path = "/mnt/yubo/emily/highknees/output/smpl/smpl/000123.json"

# === Load SMPL model (choose gender/model type as needed) ===
body_model = load_model(gender='neutral', model_type='smpl')

# === Read JSON ===
with open(json_path, "r") as f:
    data = json.load(f)[0]  # first (and only) person

poses = torch.tensor(data["poses"], dtype=torch.float32)
shapes = torch.tensor(data["shapes"], dtype=torch.float32)
Rh = torch.tensor(data["Rh"], dtype=torch.float32)
Th = torch.tensor(data["Th"], dtype=torch.float32)

# === Run SMPL model to get vertices ===
output = body_model(poses=poses, shapes=shapes, Rh=Rh, Th=Th, return_verts=True)
vertices = output["vertices"].detach().cpu().numpy().squeeze()
faces = body_model.faces

# === Visualize using trimesh (opens interactive window) ===
mesh = trimesh.Trimesh(vertices, faces)
mesh.show()