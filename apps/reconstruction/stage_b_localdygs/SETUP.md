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

## Build steps (run from a login node)

```bash
module load python/3.11 cuda gcc apptainer
virtualenv --no-download ~/envs/localdygs
source ~/envs/localdygs/bin/activate
pip install --no-index --upgrade pip

# Wheelhouse installs (Alliance prebuilt, no internet needed)
pip install --no-index numpy matplotlib opencv-python imageio
pip install --no-index torch torchvision plyfile colorama tqdm
pip install --no-index einops wandb lpips laspy torchmetrics jaxtyping
pip install --no-index pytorch-msssim torch_scatter mmcv

# tinycudann — fetch from PyPI/git (no wheelhouse build), needs setuptools<81
pip install 'setuptools<81'                           # tinycudann uses pkg_resources
cd ~/github && git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn && \
    TCNN_CUDA_ARCHITECTURES=80 \
    pip install bindings/torch                        # SM 8.0 = A100

# Apply the simple_knn patch before installing the LocalDyGS submodules
cd ~/github/LocalDyGS
git apply ~/github/Motion-Capture/apps/reconstruction/stage_b_localdygs/patches/0001-simple_knn-include-cfloat.patch
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Sanity
python -c "import torch, tinycudann, mmcv, simple_knn, diff_gaussian_rasterization; \
           print('all imports OK')"
```

You should NOT do the build on a login node for tinycudann or
the LocalDyGS submodules — they compile CUDA kernels and login nodes
have CPU-only nvcc with strict resource limits. Use a short interactive
GPU allocation:

```bash
salloc --account=rrg-vislearn --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00
# then run the build steps above
```

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

## Hardcoded values worth knowing

These live in upstream code and we may need to override later:

| File | Symbol | Value | Why we care |
|---|---|---|---|
| `scene/dataset_readers.py:198` | `test_num` | `[0,10,20,30]` | With our 11 cams (sort indices 0–10), only positions 0 and 10 become test cams (`01.jpg` and `11.jpg`). 9 train, 2 test. Acceptable for a first run. |
| `scene/dataset_readers.py:204` | `downsample` | `2.0` | LocalDyGS halves the input image side. Cow data is captured at 4K specifically *not* to be downscaled (fur texture loss). Either patch this to 1.0 or accept 2K training res for the first run. |
| `scene/dataset_readers.py:242` | `maxtime` | `300` | Hardcoded; affects positional encoding range. For shorter sequences (e.g. our 60-frame sweep) it's still fine — `time = i/N_frames` is in [0,1]. |

When we train, decide on the downsample question first. Currently
recommend overriding to 1.0 for cow data to honor the no-downscale rule.
