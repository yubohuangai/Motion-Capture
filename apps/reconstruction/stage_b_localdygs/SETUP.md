# LocalDyGS env on Narval — reproducible build recipe

LocalDyGS upstream ships a `conda` environment.yml. The Alliance cluster
disallows conda on the host, so we rebuild with `virtualenv --no-download`
against the CVMFS wheelhouse. This file is the canonical recipe.

Repo: `~/github/LocalDyGS` (upstream `WuJH2001/LocalDyGS`, ICCV 2025).

## What works (verified 2026-04-27)

- torch 2.10.0 + CUDA 12.9 (wheelhouse — newer than upstream's 2.2.0+cu118; works)
- `tinycudann` 2.0 — built at SM 8.0 (A100), needs `setuptools<81`
- `mmcv` 1.4.4 (wheelhouse — close enough to upstream's pinned 1.6.0)
- `torch_scatter` 2.1.2
- LocalDyGS-fork `diff_gaussian_rasterization` (vendored in `submodules/`)
- LocalDyGS-fork `simple_knn` (vendored, **needs the cfloat patch** — see below)

## Build steps

The CUDA-extension installs (tinycudann, simple_knn, diff_gaussian_rasterization)
need a GPU at build time so nvcc can detect compute capability. Run the
whole recipe inside a short interactive GPU allocation:

```bash
salloc --account=rrg-vislearn --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00
```

**Critical**: `module load opencv` MUST come *before* `source venv/bin/activate`,
or cv2 won't import. The opencv module sets PYTHONPATH; venv activation
freezes whatever PYTHONPATH it sees at activation time.

```bash
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
virtualenv --no-download ~/envs/localdygs
source ~/envs/localdygs/bin/activate
pip install --no-index --upgrade pip

# Wheelhouse installs (Alliance prebuilt, no internet needed)
pip install --no-index numpy matplotlib scipy imageio
pip install --no-index opencv-python-headless                   # uses opencv module
pip install --no-index torch torchvision plyfile colorama tqdm
pip install --no-index einops wandb lpips laspy torchmetrics jaxtyping
pip install --no-index pytorch-msssim torch_scatter mmcv pillow typing_extensions

# tinycudann — fetch from PyPI/git (no wheelhouse build), needs setuptools<81
pip install --no-index 'setuptools<81'                          # tinycudann uses pkg_resources
cd ~/github && git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn && \
    TCNN_CUDA_ARCHITECTURES=80 \
    pip install bindings/torch                                  # SM 8.0 = A100

# Apply all three patches before installing the LocalDyGS submodules
cd ~/github/LocalDyGS
PATCHES=~/github/Motion-Capture/apps/reconstruction/stage_b_localdygs/patches
git apply $PATCHES/0001-simple_knn-include-cfloat.patch
git apply $PATCHES/0002-dataset_readers-downsample-1.patch
git apply $PATCHES/0003-dataset_readers-test_num-bounds.patch
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Sanity (re-load modules in same one-shot, opencv must be loaded BEFORE activate)
python -c "import torch, tinycudann, mmcv, simple_knn, diff_gaussian_rasterization, cv2; \
           print('all imports OK')"
```

## Activating the env in a future session

```bash
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate
```

The opencv-before-activate ordering is the part most likely to bite you.
If `import cv2` fails, you almost certainly activated the venv first.

## Lessons (the non-obvious ones — keep these in memory)

### 1. `tinycudann` needs `setuptools<81`

`tiny-cuda-nn/bindings/torch/setup.py` imports `pkg_resources`, which
setuptools removed at v81. Pin before the install:

```bash
pip install 'setuptools<81'
```

Symptom if you forget: `ModuleNotFoundError: No module named 'pkg_resources'`
during `pip install bindings/torch`.

### 2. `simple_knn.cu` needs `#include <cfloat>` on modern toolchains

LocalDyGS's vendored `submodules/simple-knn/simple_knn.cu` uses `FLT_MAX`
without including `<cfloat>`. With CUDA 12 + gcc 11+, this header is no
longer transitively included from `<cmath>`. Symptom:

```
simple_knn.cu(NN): error: identifier "FLT_MAX" is undefined
```

Fix is at `patches/0001-simple_knn-include-cfloat.patch` in this dir —
apply with `git apply` after a fresh clone.

The vanilla 3DGS `simple-knn` repo *does* have the include — only the
LocalDyGS fork is missing it.

### 2b. `readColmapSceneInfo` hardcodes `downsample = 2.0`

Half-resolution training violates this project's no-downscale rule for
cow data (fur texture). Patch is at
`patches/0002-dataset_readers-downsample-1.patch` — flips it to 1.0.

If memory becomes a concern at native 4K (660 train images × 4K is a
lot for Gaussian splatting), reconsider, but only after measuring.

### 2c. `test_num = [0,10,20,30]` assumes >=31 cameras

Upstream `readColmapSceneInfo` hardcodes test cam indices as
`[0, 10, 20, 30]`. With smaller rigs (our cow setup is 11 cams,
indices 0..10), only 0 and 10 are valid — the test split has 2
extrinsics but `Colmap_Dataset.__init__` sets
`self.cameras = len(self.split) = 4` regardless. The off-by-N then
crashes during `load_images_path` with
`IndexError: list index out of range` at `self.poses[idx]`.

Patch is at `patches/0003-dataset_readers-test_num-bounds.patch` —
filters `test_num` to only valid indices via
`[i for i in [0, 10, 20, 30] if i < len(cam_extrinsics)]`.

Train/test split for our 11-cam rig: 9 train (cams 1..9), 2 test
(cams 0, 10). Acceptable for first runs.

### 3. `tinycudann` must be built for the right SM

A100 = SM 8.0. Set `TCNN_CUDA_ARCHITECTURES=80` *before* `pip install`,
otherwise tinycudann auto-detects from the build node's GPU (which is
"none" on a login node and "wrong" on a non-A100 dev box).

### 4. Use the wheelhouse `mmcv==1.4.4`, not pinned 1.6.0

Upstream pins `mmcv==1.6.0`. The wheelhouse only has 1.4.4. The two
versions have a compatible API for what LocalDyGS actually uses
(`Config` loader, registry). Don't try to build 1.6.0 from source on
Narval — long compile, fragile.

### 5. Verify the env before training

After install, additionally check the dataset reader imports cleanly:

```bash
cd ~/github/LocalDyGS
python -c "from scene.dataset_readers import readColmapSceneInfo; print('reader OK')"
```

This catches missing `cv2`, `colorama`, `plyfile` etc. that LocalDyGS
imports lazily.

### 6. Do NOT install `open3d` into this venv — use `~/envs/cleanply/` for prep

`prepare_localdygs_data.py` needs `open3d` to voxel-downsample the init
point cloud, but **don't** install open3d into `~/envs/localdygs/`. The
wheelhouse `open3d-0.19.0+computecanada` package pins `torch==2.6.0`,
which silently downgrades torch from 2.10 → 2.6 and ABI-breaks the
CUDA extensions (`simple_knn`, `tinycudann`, `diff_gaussian_rasterization`)
that were built against torch 2.10. Recovering takes a `pip install
--force-reinstall torch==2.10.0` plus uninstalling open3d.

Instead, run the prep script under the dedicated lightweight
`~/envs/cleanply/` venv (numpy + open3d + plyfile, no torch). The
`run_localdygs_prep.sh` SLURM script already does this. Training and
rendering still use `~/envs/localdygs/`.

### 7. Pre-cache pretrained model weights on a login node

LocalDyGS's `train.py` imports `lpips` at module scope, and
`lpips.LPIPS(net='vgg')` downloads two things from the internet:

1. **VGG16 backbone** (`vgg16-397923af.pth`, ~528 MB) → torchvision pulls
   from `download.pytorch.org` into `~/.cache/torch/hub/checkpoints/`.
2. **LPIPS calibration weights** (`vgg.pth`, ~7 KB) → the wheelhouse
   build of `lpips==0.1.4+computecanada` is **missing** these bundled
   data files. Stock pip-from-PyPI lpips includes them, but Compute
   Canada's wheel does not.

Compute nodes have no internet. If either is missing on the compute node,
LocalDyGS hangs ~9 min on TCP retry then crashes with
`urllib.error.URLError: [Errno 101] Network is unreachable`.

**Pre-cache once on a login node** before submitting any training job:

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate

# 1. VGG16 backbone (downloads from torchvision)
python -c "import lpips; lpips.LPIPS(net='vgg')"   # initially fails — see #2

# 2. LPIPS bundled calibration weights (fetch from upstream GitHub)
DEST=$(python -c "import lpips, os; print(os.path.dirname(lpips.__file__))")/weights/v0.1
mkdir -p "$DEST"
for net in vgg alex squeeze; do
    wget -O "$DEST/${net}.pth" \
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/${net}.pth"
done

# 3. Verify
python -c "import lpips; m = lpips.LPIPS(net='vgg'); print('LPIPS init OK')"
```

`run_localdygs_smoke.sh` has a fail-fast pre-flight that checks both
files exist before invoking `train.py` — if you see "required pretrained
weight missing" in the smoke job log, run the steps above on a login
node and resubmit.

## Hardcoded values worth knowing

These live in upstream code and we may need to override later:

| File | Symbol | Value | Why we care |
|---|---|---|---|
| `scene/dataset_readers.py:198` | `test_num` | `[0,10,20,30]` | With our 11 cams (sort indices 0–10), only positions 0 and 10 become test cams (`01.jpg` and `11.jpg`). 9 train, 2 test. Acceptable for a first run. |
| `scene/dataset_readers.py:204` | `downsample` | `2.0` | LocalDyGS halves the input image side. Patched to 1.0 in `patches/0002-*` to honor cow no-downscale rule. |
| `scene/dataset_readers.py:242` | `maxtime` | `300` | Hardcoded; affects positional encoding range. For shorter sequences (e.g. our 60-frame sweep) it's still fine — `time = i/N_frames` is in [0,1]. |

The `downsample` question is now resolved (patch 0002 sets it to 1.0).
The other two are runtime hyperparameters that don't currently block us.
