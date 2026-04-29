# Reconstruction Project — Live Status

> **Purpose**: single source of truth for "where are we *right now*". Updated
> after every completed unit of work and committed to git. If a Claude session
> dies (bus error etc.), the next session reads this file and continues
> without losing context.
>
> **Architectural decisions, pipeline overview, design rationale** → `PROJECT.md`.
> **Live progress, last completed step, next command** → this file.

---

## Current state

**On Rorqual (H100-80G).** Migration from Narval is functionally
complete. As of 2026-04-29 morning:

| Track | Status |
|---|---|
| Code repos (Motion-Capture, LocalDyGS, tiny-cuda-nn, yubo-brain) | ✅ cloned on Rorqual |
| 6 LocalDyGS patches applied | ✅ done |
| Venvs (`localdygs/`, `cleanply/`, `globus-cli/`) | ✅ built on Rorqual |
| `globus-cli` install + login + data_access consent | ✅ done both clusters |
| Globus transfer: raw `cow_1/9148_10581/` (27 GB on disk) | ✅ done (task `59078659-...`) |
| Globus transfer: Stage 1 `stage_a/colmap_4d/` (189 GB on disk) | ✅ done (task `59d80354-...`) |
| Globus transfer: Stage 2 `train_planB_e3/` (6.1 GB on disk) | ✅ done (task `5a79c04e-...`) |
| Phase 1 — H100 verify (full H100 import + tinycudann SM 9.0 fwd) | ✅ done (job `11093921`) |
| Phase 3 — `export_3dgs_ply.py` + 3 cow PLYs at t={0, 0.5, 1.0} | ✅ done (job `11097919`, 1:46) — see below |
| Render R1 + R2 on Narval (Plan B Exp 1, Exp 2) | ⚠️ status unknown — last known in-flight on Narval |
| Phase 2 — Globus transfer e1 (best model) | ✅ landed; PLY export auto-triggered (`11098033`) |
| Phase 2 — Globus transfer e2 | 🔄 in flight |
| Phase 4a (Exp 4) — bigger init pcd 250K, **no** mask-aware, 136 fr | ⏳ prep `11098010` ✅ → train `11098011` PD |
| Phase 4a' (Exp 5) — bigger init pcd 250K + **mask-aware**, 136 fr | ❌ skipped tonight — index-collision with Stage 1 partials made apples-to-apples-with-Exp1 prep impractical; user can run later |
| Phase 4b (Exp 6) — stride-5 over full seq (286 fr), mask-aware, 60K iters | ⏳ Stage 1 array `11098093` (150 missing fr; 2 running, 148 pending; orchestrator waits) → undistort + prep_e6 + train_e6 chained |
| Phase 5 — trajectory extractor on e3 (test) | ⏳ `11098019` PD |
| Phase 5 — trajectory extractor on **e1 (best model)** + cluster PLYs at K∈{4,8,16} | ⏳ `11098174` extract → `11098175` cluster |
| Bug — undistort_masks symlinks `.png` instead of `.jpg` | 🐛 → ✅ fixed in `38159ca`; verified `dense_masks/images/` populated with 11 files for all 136 frames |
| Bug — calibration `intri.yml`/`extri.yml` symlinks broken on Rorqual (point to `cow_1/10465/`) | 🐛 → ✅ transferred `cow_1/10465/` (task `cfb586ba-...`); recreated symlinks; Stage 1 array re-submitted as `11098093` |

What's NOT on Rorqual yet (deliberate, for now):
- `train_planB_e1` (best 136-fr mask-aware model; test PSNR 12.14) and `train_planB_e2`
- `scene_planB_e3` and other LocalDyGS scene-prep dirs (the Stage 1 → 2 prep output;
  not needed for the export script which loads `GaussianModel` directly without `Scene`)

**Next decisions (Phase 2 / Phase 4)**: see open questions below.

### Phase 3 results — exported PLYs ready to view

`/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e3/exported/`:

| t | active gauss. | kept (AABB) | file |
|---|---|---|---|
| 0.00 | 545,585 | 543,201 (99.6%) | `cow_t000.ply` 128.5 MB |
| 0.50 | 490,006 | 487,649 (99.5%) | `cow_t050.ply` 115.3 MB |
| 1.00 | 563,477 | 561,037 (99.6%) | `cow_t100.ply` 132.7 MB |

PLY uses the standard 3DGS schema (62 properties incl. SH degree 3
with 45 zero `f_rest` fields), binary little-endian, plyfile-verified.
Opacity / scale ranges sane. Drop these into SuperSplat
(https://playcanvas.com/supersplat/editor) to rotate around the cow.

Caveats:
- Trained on Plan B Exp 3 (60 frames stride-5 over 0..295, mask-aware loss);
  test PSNR was 11.82, not the best model. The best model is `train_planB_e1`
  (136 frames + mask-aware, test PSNR 12.14) — currently still on Narval.
- Coordinate frame is Colmap (Y-down). SuperSplat is Y-up, so the cow may
  display upside-down at first; rotate in the viewer.
- File sizes are ~120 MB (full SH-degree-3 schema). If you prefer smaller
  PLYs (~35 MB), I can add a `--no-sh-rest` flag that drops `f_rest_*`.

## Plan: continue on Rorqual

After R1 / R2 finish on Narval and the in-flight Globus transfers
complete, the resumption checklist:

### Phase 1 — verify the migration landed cleanly (~15 min on Rorqual)

```bash
# 1. Pull latest yubo-brain (PROJECT.md updates, the new install-claude-md.sh, etc.)
cd ~/github/yubo-brain && git pull
cd ~/github/Motion-Capture && git pull

# 2. Confirm transferred data is on /scratch
du -sh /scratch/yubo/cow_1/9148_10581/ \
       /scratch/yubo/cow_1/9148_10581_output/stage_a/ \
       /scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e3/

# 3. Smoke-test localdygs venv on a GPU node (verify CUDA exts work on H100)
salloc --account=rrg-vislearn --gres=gpu:h100_3g.40gb:1 --cpus-per-task=4 --mem=16G --time=0:30:00
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate
python -c "
import torch, simple_knn, tinycudann, diff_gaussian_rasterization, cv2
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('CUDA exts: simple_knn, tinycudann, dgr, cv2 — all import OK')
"
```

If tinycudann fails with "Unknown compute capability", it was built for
SM 8.0 (A100) and needs a rebuild for SM 9.0 (H100). Per
`apps/reconstruction/stage_b_localdygs/SETUP.md`:

```bash
cd ~/github/tiny-cuda-nn
TCNN_CUDA_ARCHITECTURES=90 pip install bindings/torch --force-reinstall
```

### Phase 2 — fetch the e1 / e2 trained models from Narval if you want them

Once R1 + R2 finish on Narval (~1 h after this STATUS.md was written),
add a Globus transfer for the new outputs:

```bash
# (on rorqual2 or Narval — globus is server-side)
NARVAL=a1713da6-098f-40e6-b3aa-034efe8b6e5b
RORQUAL=f19f13f5-5553-40e3-ba30-6c151b9d35d4

# train_planB_e1 = best model (test PSNR 12.14)
globus transfer \
    $NARVAL:/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e1/ \
    $RORQUAL:/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e1/ \
    --recursive --label "train_planB_e1 (post-R1)"

globus transfer \
    $NARVAL:/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e2/ \
    $RORQUAL:/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e2/ \
    --recursive --label "train_planB_e2 (post-R2)"
```

If you'd rather just retrain on H100 (faster), skip — see Phase 4.

### Phase 3 — interactive 3D viewer (cow only, no scene)

User-facing deliverable: a standalone `.ply` you can open in
[SuperSplat](https://playcanvas.com/supersplat/editor) (browser) or
any 3DGS viewer to rotate around the cow.

This needs a **converter** from LocalDyGS's anchor-based representation
to standard 3DGS PLY format at a chosen time `t ∈ [0, 1]`. To be
written as
`apps/reconstruction/stage_b_localdygs/export_3dgs_ply.py`. Logic:

1. Load LocalDyGS scene + `iteration_30000` checkpoint (anchor +
   FDHash + MLPs + time embedding).
2. For chosen time `t`, replicate LocalDyGS's render forward pass to
   compute per-anchor active Gaussians: position, scale, rotation,
   opacity, RGB.
3. Filter to "cow only" via the init-pcd bounding box (since
   `position_lr=0`, anchors are fixed at their cow-only init pcd
   locations — likely already cow-only, but bbox-clip for safety).
4. Convert to standard 3DGS PLY schema:
   - x, y, z (position)
   - nx, ny, nz (zeros)
   - f_dc_0..2 (RGB at SH order 0)
   - f_rest_0..44 (zeros)
   - opacity (logit-space)
   - scale_0..2 (log-space)
   - rot_0..3 (quaternion)
5. Write binary little-endian PLY for SuperSplat compatibility.

Reference: `~/github/gaussian-splatting/scene/gaussian_model.py:save_ply`
for the target schema. Estimated ~1 evening of work; can be done on
Rorqual after Phase 1 verifies.

Output: `train_planB_e1/exported/cow_t<TT>.ply` for several `t` values.

### Phase 4 — improvement experiments (see "How results would improve" below)

Pick one or more based on what's interesting:

- **Stride-1 over the full sequence** = 1434 frames × 9 cams = 12,906
  train images. Iters need to scale: 30K → ~150K. Wall on H100: ~10 h.
  Biggest expected improvement (more angular coverage of cow).
- **Stride-5 over the full sequence** = 287 frames; 60K iters; ~4 h
  on H100. Sweet spot — 5× more data than current 136-fr.
- **Bigger init pcd** (`--target-points 250000`): cheap, may help
  spatial coverage for fine cow features (face, legs).
- **Enable densification** (lower `start_stat = 1500000` to e.g.
  `5000`): lets model spawn new Gaussians during training; risk of
  memory growth.
- **Fairer test split** (cams 04 + 09 instead of 01 + 11): re-prep +
  retrain so test PSNR is interpretable as actual generalization.

### Phase 5 — Stage 3 (articulation discovery, research)

Once a model is in good shape, start extracting per-Gaussian
trajectories and clustering them into rigid parts. See PROJECT.md §6
"Stage 3 (deferred)" for the breakdown. This is the project's actual
research contribution.

## How results would improve (Yubo's question)

**More frames over the full sequence is the biggest lever.** The
current 136 frames span the cow's full 48 s of motion at mixed stride
(stride-5 in first 10 s, stride-15 thereafter). Increasing density:

| Frame strategy | Train imgs | Iters needed | H100 wall | Expected gain |
|---|---|---|---|---|
| Current 136 (mixed) | 1224 | 30K | ~1 h | (baseline: test PSNR 12.14) |
| Stride-15 over 0..1425 (96 fr, uniform) | 864 | 30K | ~50 min | ~same as current |
| **Stride-5 over 0..1425** (287 fr) | 2583 | 60K | ~4 h | **+1 to +2 PSNR** likely |
| Stride-1 over 0..1434 (1434 fr) | 12906 | 150K | ~10 h | +1 to +2 over stride-5 (diminishing) |

Smaller stride helps mostly because the cow's motion isn't uniform
(per Yubo: stops, walks, rotates at varying speeds). With stride-15
some rotation transitions are crossed in a single jump, leaving the
deformation MLP to interpolate across larger time gaps. Stride-5 over
the full range gives consistent ~6 fps temporal sampling throughout
the rotation.

**Other levers** (ranked by my expected impact, from yubo-brain
[[3dgs-scaffold-fixed-anchors]] and the cow_1 experiments):

1. **More frames at finer stride** ⭐ (above table)
2. **Bigger init pcd** (250K points instead of 90K). LocalDyGS's
   `position_lr = 0` design means anchors are fixed at init pcd
   locations and never move. More anchors = finer surface coverage
   = potentially sharper cow detail. Cheap to try (re-prep is 30 s).
3. **Enable densification** (lower `start_stat` from 1.5M to ~5K).
   Lets the model spawn new Gaussians where the loss says it needs
   more capacity. Risk: memory growth at 4K resolution.
4. **More iterations at same data** (60K instead of 30K). Gradual
   improvement, diminishing returns past ~80K typically.
5. **Better test split** (cams 04 + 09 instead of extremes 01 + 11)
   — doesn't improve the model, but makes the test PSNR
   interpretable as actual generalization, not "extreme extrapolation."

The user's intuition (more images, smaller stride) aligns with #1
exactly. That's the right first experiment on Rorqual.

## Plan A / Plan B results so far (durable record)

Test PSNR is generalization to held-out cams 01 + 11 (rig extremes —
unfair "extremes" split; see [[3dgs-train-test-evaluation-semantics]]).
Train PSNR is fit to the training cams.

| Run | Frames | Loss | Train PSNR | Test PSNR | Output dir |
|---|---|---|---|---|---|
| Plan A | 60 (stride-5 over 0..295) | standard | 22.94 | 11.46 | `train_20260427_220326/` |
| Plan B Exp 3 | same 60 | mask-aware | 23.09 | 11.82 | `train_planB_e3/` |
| Plan B Exp 2 | 136 (stride-mixed full rot.) | standard | 22.49 | 11.99 | `train_planB_e2/` |
| **Plan B Exp 1** | **136** | **mask-aware** | 22.49 | **12.14** ⭐ | `train_planB_e1/` |

All checkpoints at iter 30000 with the same basketball.py preset.
Mask-aware loss + full-rotation frames compound; both interventions
help.

## Recent activity (newest first)

| Date | Event |
|---|---|
| 2026-04-29 | **Overnight push** — chained pipelines for e4 (bigger pcd no-mask 136fr) and e6 (stride-5 + mask, 286fr, 60K iters); plus Phase 5 trajectory extraction + cluster baseline on best model e1; e1 PLY export auto-fired. **Two latent bugs found + fixed**: (1) undistort_masks symlinked masks as `.png` but COLMAP looks for `.jpg` → `dense_masks/images/` empty for ALL 136 frames since the original Narval run → mask-aware loss silently fell back to standard L1+SSIM (suspected to invalidate the +0.36/+0.68 dB attributed to Exp 1/3 vs Plan A — see warning section below). Fix in `38159ca`. (2) `intri.yml`/`extri.yml` were symlinks to `cow_1/10465/` which wasn't transferred → Stage 1 array crashed at `assert os.path.exists(intri_name)`. Fix: Globus-transferred `cow_1/10465/` and recreated the calibration symlinks. New `extract_trajectories.py` (per-Gaussian center over t∈[0,1]) and `cluster_trajectories.py` (k-means baseline, colored static PLY) added as Stage 3 starters. |
| 2026-04-29 | **Phase 3 deliverable shipped** — `cow_t{000,050,100}.ply` (115–133 MB each, 487K–561K Gaussians) at `train_planB_e3/exported/`, ready for SuperSplat. PLYs verified with plyfile, schema is standard 3DGS SH-deg-3, opacity/scale ranges sane. |
| 2026-04-29 | Phase 1 verified on Rorqual: data sizes correct (27 / 189 / 6.1 GB), localdygs venv imports OK on H100 (job `11093921`, last session), all 3 Globus transfers SUCCEEDED. Phase 3 exporter `apps/reconstruction/stage_b_localdygs/export_3dgs_ply.py` written; submitted as job `11097919` for first test (cow_t000/050/100.ply at iter 30000). |
| 2026-04-28 | PROJECT.md cleanup: stage naming consistency (Stage 0 / 1.x / 2.x), sparse-masking description corrected (`auto` policy default, not always-unmasked) |
| 2026-04-28 | yubo-brain: globus-cli-on-alliance cheatsheet committed; install-claude-md.sh made cluster-aware; "Onboard a new machine" workflow added; Motion-Capture lessons ingested as 8 new wiki pages |
| 2026-04-28 | Rorqual onboarded via yubo-brain workflow; per-machine canonical at `wiki/claude-code-rorqual-instructions.md` |
| 2026-04-28 | Globus transfers: raw + stage_a + train_planB_e3 in flight from Narval to Rorqual |
| 2026-04-28 | T1, T2 (136-frame Plan B) completed: best test PSNR 12.14 (Exp 1, mask-aware). R1 + R2 (renders) had hardcoded-frames bug, fixed `9c5bd02`, resubmitted with FRAMES_END=136 + 3h walltime |
| 2026-04-28 | Plan B chain submitted (Stage 1 array → undistorts → 3 prep → 3 train → 3 render); Exp 3 fully completed (train PSNR 23.09 / test PSNR 11.82) |
| 2026-04-27 | Plan A completed end-to-end (60-frame, no mask-aware): train PSNR 22.94 / test PSNR 11.46 |
| 2026-04-27 | Patch 0006 (mask-aware loss) landed; postprocess_render.py written |
| 2026-04-27 | Patches 0001-0005 landed (cfloat, downsample, test_num bounds, render GPU, render incremental save) |
| 2026-04-27 | LocalDyGS env built; STATUS.md introduced |
| 2026-04-26 | Stage 1 60-frame stride-5 sweep completed |
| 2026-04-26 | Initial PROJECT.md draft |

## Wake-up helper (Narval-side)

`scripts/check_planB_status.sh` — prints state of all Plan B SLURM
jobs. Useful if you re-open a Narval session before the migration is
fully done.

## ⚠️ Mask-aware loss results may be invalid — investigation needed

The undistort_masks bug discovered tonight (`.png` filenames vs sparse
model's `.jpg` references — see `38159ca`) means COLMAP image_undistorter
silently failed to write `dense_masks/images/` for ALL 136 frames during
the original Narval run. Patch 0006's logic when `<scene>/frame<N>/masks/`
exists-but-is-empty falls back to None mask → standard L1+SSIM loss.

If this also held on Narval at training time (likely — the bug pattern
matches the .log files transferred over, where Narval also got
"Cannot read image at path .../01.jpg"), then:

- Plan B Exp 1 ("136 fr + mask-aware", test 12.14) — actually no-mask
- Plan B Exp 2 (136 fr no-mask, test 11.99)
- Plan B Exp 3 ("60 fr + mask-aware", test 11.82) — actually no-mask
- Plan A (60 fr no-mask, test 11.46)

Then Exp 1 ≈ Exp 2 (both 136 fr no-mask) and the 0.15 dB difference is
just init/seed variance or a config detail I haven't spotted. The
"+0.36 dB mask-aware help" attributed to Exp 3 vs Plan A would also be
unrelated to mask-aware.

**e5 and e6 (queued tonight) are the first TRUE mask-aware runs.** If
they outperform e4 / Exp 1 / Exp 2 by a meaningful margin, we'll have
actual mask-aware evidence. If they don't, the masking lever may simply
not help on the cow-on-uniform-floor data (the no-mask loss was already
focusing on cow signal because the floor is roughly constant
brightness).

To confirm, after e5 finishes: spot-check that
`scene_planB_e5/frame000000/masks/` actually contains 11 PNG files (not
empty) — if so, mask-aware was genuinely active for that run.

## Open questions to resolve eventually

- Visual quality of the trained models — pending Phase 3 viewer or
  pending Phase 1 inspection of existing renders on Mac.
- Stage 1 work/ contents are 200 GB; ~95% is `dense/` PatchMatch
  state that's not used downstream. Could `rm -rf
  /scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d/work/frame_*/dense/`
  on Rorqual after data lands to save 160+ GB.
- Whether to rename `stage_a_colmap_4d/` → `stage_a_colmap_perframe/`
  (PROJECT.md §9 cleanup TODO). Defer until other work settles.
