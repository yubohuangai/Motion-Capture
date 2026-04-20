# Reconstruction Classical — Stage A & Stage B Explained

A detailed walk-through of what the two stages of `apps/reconstruction_classical/` actually do, what knobs control them, and why they can produce garbage.

---

## Big picture

```
posed images (intri.yml + extri.yml + images/*.jpg)
                │
                ▼
      ┌──────────────────┐
      │   STAGE A        │    Classical Multi-View Stereo
      │   (CPU, ~minutes)│
      │                  │
      │  sparse  →  MVS  │    fused.ply (dense point cloud)
      │           ↓      │    mesh.ply  (Poisson mesh)
      │         fusion   │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │   STAGE B        │    Neural Implicit Surface (NeuS)
      │   (GPU, ~1 hour) │
      │                  │    mesh_neus.ply (from SDF + marching cubes)
      └──────────────────┘
```

Stage A gives you a geometric, hand-engineered result in minutes. Stage B slowly optimizes a neural SDF per scene for a smoother, more complete surface.

---

## Stage A — Classical MVS

Stage A has three sub-stages: **sparse → MVS → fusion**, plus a Poisson meshing step at the end.

### A.1 Sparse reconstruction (SIFT + matching + triangulation)

File: `stage_a_classical/sparse.py`

Goal: produce a small, high-precision seed point cloud used to (a) set depth ranges for MVS and (b) provide scene bounds for NeuS.

Pipeline:

1. **SIFT detection** on each downscaled image (`--sparse_downscale 0.5`). Uses OpenCV SIFT with `contrastThreshold=0.02`. You saw ~13k–25k keypoints per view in your run.
2. **Pairwise matching** for all `C(N, 2)` camera pairs using FLANN KD-tree. Lowe's ratio test at `0.75` + mutual-nearest-neighbor check.
3. **Epipolar filter**: uses your known poses to compute the fundamental matrix per pair, then rejects matches farther than `--max_epi_px 2.0` px from their epipolar line. (This is what turned `match 07-08: raw=1763` into `kept=1650` in your log.)
4. **Track building** (union-find): merge pairwise matches into multi-view tracks. Tracks with `< min_views=3` observations are dropped.
5. **Multi-view DLT triangulation** per track, rejecting points that go behind any camera or reproject with error `> --max_reproj_err 2.5` px.

Output: `sparse.ply`. Your run produced 1803 points with median reproj 1.33 px — that's healthy.

**Why it can be bad**: textureless scenes (blank walls, shiny cow fur under uniform light) give too few keypoints; wrong calibration breaks the epipolar filter so *everything* gets rejected.

### A.2 Dense MVS (plane-sweep stereo)

File: `stage_a_classical/mvs_plane_sweep.py`

Goal: compute a per-pixel depth map for each view.

Plane-sweep in one paragraph: for a given reference view, hypothesize a set of fronto-parallel depth planes. For each plane, warp all source views onto the reference using the induced homography, compare warped patches to the reference with ZNCC (Zero-mean Normalized Cross-Correlation), and pick the depth that matches best per pixel.

Key knobs and behavior:

- **Depth range**: auto-estimated from the sparse cloud — 2nd/98th percentile of in-frustum sparse depths with log-padding `pad_ratio=0.25`. This is why sparse feeds MVS.
- **Plane sampling**: `--n_depths 128` planes, uniform in **inverse depth** (better near-camera resolution).
- **Source view selection**: up to `--max_sources 4` cams per reference. A source must have a baseline angle between **5° and 45°** to the reference; higher score = forward-axis similarity × baseline.
- **Matching cost**: ZNCC over a `--ncc_ksize 7`×7 window, aggregated across sources as either mean or `top2` best.
- **Confidence**: how "peaky" the cost-vs-depth curve is (best plane vs. average). Low confidence → pixel rejected at `--min_conf 0.25`.
- **Low-texture reject**: variance < `2.0` → invalid.
- **Post-processing**: median filter, joint bilateral filter for edge-preserving smoothing.

### ⚠️ This is where your run got the first real problem

Look at your log:

```
[mvs] ref=11 srcs=[]; skipping
[mvs] ref=01 srcs=['09', '08', '10', '07']
[mvs] ref=02 srcs=['03', '05', '04', '06']
```

Two things:

1. **Camera 11 got zero sources** — it's at the end of the half-circle and its 5°–45° baseline window didn't include any other cam.
2. **Camera 01 picked `09, 08, 10, 07`** — but physically 01 sits between **06 and 07**. It should see 06 and 07 as nearest neighbors. The source-view selection is picking by optical-axis alignment, which for a half-circle rig can give wrong neighbors when the pose optimization hasn't fully converged.

This correlates with your sparse matching matrix: `01-06: kept=84`, `01-07: kept=1950`, `01-08: kept=673`. Camera 01 is matching strongly with 07/08 but weakly with 06 — suggesting the **calibration of 01 or 06 is off**, or their physical position is not what the template says.

### A.3 Fusion (cross-view consistency + cleanup)

File: `stage_a_classical/fusion.py`

Goal: aggregate per-view depth maps into one clean world-space point cloud.

1. **Back-project** each depth map to 3D; estimate per-point normals from depth gradients.
2. **Consistency check**: project each 3D point into every other view and require depth agreement within `--rel_tol 0.02` (relative) in at least `--min_consistent 2` other views. This kills outliers.
3. **Voxel downsample** at `--voxel_size 0.01` m.
4. **Statistical outlier removal**: k-NN (k=16) distances within `std_ratio=2.0` σ.

Output: `fused.ply`. Your run: 142k → 128k → 123k pts — looks fine *numerically*, but garbage in = garbage out from the MVS step.

### A.4 Poisson meshing

File: `stage_a_classical/poisson.py`

Screened Poisson surface reconstruction at octree `--poisson_depth 9`. Prunes vertices in the bottom `--poisson_density_pct 5.0` percentile of point-density (removes "balloon" artifacts).

### ⚠️ Your second real problem

```
[ERROR] Failed to close loop [8: 335 288 311] ...
```

Poisson needs **consistent, well-oriented normals** on a roughly watertight point cloud. "Failed to close loop" means the normal field has sign flips or the cloud has large gaps. Common causes:

- **Half-circle rig → the back of the object is never seen** → Poisson can't close the surface. Expected.
- Noisy MVS depth → normals flipping around edges.

The error is often printed but the mesh still gets saved; usually just with holes and bad topology on the unseen side. That's **fundamentally unavoidable** with only a half-circle; the object's back must be cropped or you accept the hole.

---

## Stage B — NeuS (neural implicit surface)

Files: `stage_b_neus/train.py`, `models.py`, `renderer.py`

### What NeuS is

NeuS (Wang et al., NeurIPS 2021) represents the scene as the zero-level-set of a **learned signed distance function**, then renders it with NeRF-style volume rendering. Advantage over NeRF for reconstruction: the SDF's zero-crossing is a crisp, single surface — much better geometry than raw NeRF density.

Two small MLPs are learned per scene from scratch:

- **SDFNetwork** — 8 layers, 256 hidden. Input: 3D position (with 6-band positional encoding). Output: (SDF scalar, 256-d feature). Initialized so that SDF ≈ distance to a unit sphere — this "geometric init" is critical for stable training.
- **ColorNetwork** — 4 layers. Input: (SDF feature, viewing direction, normal). Output: RGB.
- **SingleVariance** — one learnable scalar controlling the softness of the level-set.

### Training loop

1. Scene bounds come from `sparse.ply` (Stage A) — every point is scaled into a unit cube (`--bound_padding 1.1`).
2. Each iteration: sample `--batch_rays 512` random pixels across all training images, render each ray via volume rendering over `--n_samples 64` coarse + `--n_importance 64` fine samples.
3. Loss = photometric MSE + `--weight_eikonal 0.1` × Eikonal penalty (`‖∇SDF‖ → 1`). Eikonal keeps the SDF a true distance field.
4. Adam, cosine-annealed LR starting at `--lr 5e-4`, `--n_iters 100_000`.
5. Every 5k iters: render a validation view to `val/`, save a checkpoint.

### Extracting the mesh

After training: evaluate SDF on a `--mesh_resolution 256`³ grid inside the scene bbox, run marching cubes at level 0. Output: `mesh_neus.ply`.

### Why NeuS can still fail on your data

- **Wrong scene bounds** — if `sparse.ply` is garbage, the unit cube doesn't contain the object.
- **Bad poses** — NeuS has no pose refinement; pose errors show up as floaters.
- **Insufficient angular coverage** — half-circle rig means the SDF has no supervision on the back; the mesh will be thin/collapsed there.
- **Eikonal too weak** — SDF collapses to noise. Too strong → over-smoothed blob.

---

## Your specific run — diagnosis checklist

Based on the log you showed:

| Symptom | Likely cause | Action |
|---|---|---|
| Cam 11 skipped | Half-circle endpoint, no source within 5–45° baseline | Accept it (end-of-arc views always lose), OR widen `--max_angle` in `pick_source_views` |
| Cam 01's sources are `09,08,10,07` instead of `06,07` | Pose of cam 01 is misaligned; it's "looking" parallel to 07–10 more than 06–07 | **Re-run calibration.** Suspect cam 01's extrinsic — it's the odd-one-out in the half-circle template |
| Poisson "failed to close loop" | Half-circle rig has no back-side coverage | Either (a) crop the point cloud to a bbox, (b) skip Poisson and rely on NeuS mesh, (c) accept open mesh |
| Few matches 01↔06 (kept=84) vs 01↔07 (kept=1950) | Cam 01/06 relative pose is wrong | **Calibration issue**, not a reconstruction issue |

**Bottom line**: your garbage is almost certainly upstream — calibration of camera 01 doesn't agree with its stated position between 06 and 07. Before tweaking reconstruction knobs, verify the extrinsics, e.g. project a few chessboard corners from cam 01 using cams 06 and 07's triangulated 3D and see if the reprojection lands on the actual corners.

---

## Useful flags cheat-sheet

### Stage A (`run_stage_a.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--downscale` | 1.0 | downscale images for MVS |
| `--sparse_downscale` | 0.5 | downscale for SIFT |
| `--n_depths` | 128 | plane-sweep depth hypotheses |
| `--max_sources` | 4 | src views per reference |
| `--ncc_ksize` | 7 | ZNCC window |
| `--min_conf` | 0.25 | MVS confidence threshold |
| `--rel_tol` | 0.02 | fusion depth agreement |
| `--min_consistent` | 2 | fusion min consistent views |
| `--voxel_size` | 0.01 | fusion voxel size (m) |
| `--poisson_depth` | 9 | Poisson octree depth |
| `--poisson_density_pct` | 5.0 | prune low-density verts |

### Stage B (`run_stage_b.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--n_iters` | 100000 | training steps |
| `--batch_rays` | 512 | rays per step |
| `--n_samples` / `--n_importance` | 64 / 64 | coarse / fine samples |
| `--weight_eikonal` | 0.1 | SDF regularization |
| `--bound_padding` | 1.1 | bbox expansion factor |
| `--mesh_resolution` | 256 | marching cubes grid |
| `--lr` | 5e-4 | Adam learning rate |
