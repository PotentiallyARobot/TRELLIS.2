# ============================================================
# 🔺 TRELLIS.2 — Single Image Processor (Safe Sequential Mode)
#
# Upload 1 or more images. Each is fully processed before the
# next one starts. No threads, no parallelism, no VRAM contention.
#
#   For each image:
#     1. Reconstruct (pipe.run)
#     2. Render snapshot (optional)
#     3. Offload models → CPU
#     4. Prepare mesh (remesh/simplify/BVH)
#     5. xatlas UV unwrap
#     6. Texture bake + GLB export
#     7. Full cleanup
#     8. Reload models → GPU (if more images)
#
# ============================================================
import os, sys
sys.path.append("/content/TRELLIS.2")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# ── Upload ──
from google.colab import files as colab_files
import pathlib, shutil

print("📂 Upload your image(s):")
uploaded = colab_files.upload()
if not uploaded:
    raise SystemExit("No files uploaded.")

INPUT_DIR = pathlib.Path("/content/images_in")
if INPUT_DIR.exists():
    shutil.rmtree(INPUT_DIR)
INPUT_DIR.mkdir()
# Write from upload bytes — strip Colab's " (1)" duplicate suffix
import re
for name, data in uploaded.items():
    clean = re.sub(r'\s*\(\d+\)(?=\.\w+$)', '', pathlib.Path(name).name)
    (INPUT_DIR / clean).write_bytes(data)
print(f"✅ {len(uploaded)} image(s) uploaded")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

PRESET = "max_quality"   # "max_quality" | "balanced" | "fast"

PRESETS = {
    "max_quality": dict(
        ss_steps=12, shape_steps=12, tex_steps=12,
        texture_size=4096, decimate_target=1_000_000,
        remesh=True, remesh_band=1.0,
        render_mode="snapshot",   # "video" | "snapshot" | "none"
        video_resolution=512,
    ),
    "balanced": dict(
        ss_steps=8, shape_steps=8, tex_steps=8,
        texture_size=2048, decimate_target=500_000,
        remesh=True, remesh_band=1.0,
        render_mode="snapshot",
        video_resolution=512,
    ),
    "fast": dict(
        ss_steps=8, shape_steps=8, tex_steps=8,
        texture_size=2048, decimate_target=500_000,
        remesh=False, remesh_band=1.0,
        render_mode="none",
        video_resolution=512,
    ),
}

CFG = PRESETS[PRESET]
OUTPUT_DIR = pathlib.Path("/content/drive/MyDrive/trellis_batch_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════

import time, re, gc, traceback
import torch, numpy as np
from PIL import Image
import cv2, imageio

os.environ["TRELLIS2_DISABLE_REMBG"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_DIR = pathlib.Path("/content/TRELLIS.2")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

torch.set_float32_matmul_precision("high")

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9

print("=" * 60)
print(f" GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM:.1f} GB")
print(f" Preset: {PRESET}")
print("=" * 60)

import postprocess_parallel as pp

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def safe_stem(name):
    s = pathlib.Path(name).stem.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "image"

def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass  # CUDA context may be poisoned — skip silently
    gc.collect()

def vram_free():
    return TOTAL_VRAM - torch.cuda.memory_allocated() / 1e9

def cuda_ok():
    """Check if CUDA context is still healthy."""
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False

def safe_offload():
    """Move models to CPU, tolerating a poisoned CUDA context."""
    for _, model in pipe.models.items():
        try:
            model.to("cpu")
        except RuntimeError:
            pass
    try:
        pipe.image_cond_model.to("cpu")
    except RuntimeError:
        pass
    cleanup()

# Max faces for render — above this the nvdiffrec renderer can trigger
# illegal memory access which poisons the entire CUDA context.
RENDER_MAX_FACES = 16_000_000

# ══════════════════════════════════════════════════════════════
# LOAD PIPELINE
# ══════════════════════════════════════════════════════════════

print("\n🔧 Loading pipeline (low_vram)...")
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.low_vram = True
pipe.cuda()

hdri = REPO_DIR / "assets" / "hdri" / "forest.exr"
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread(str(hdri), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device="cuda",
))
print(f"✅ Pipeline loaded | alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ══════════════════════════════════════════════════════════════
# GATHER IMAGES
# ══════════════════════════════════════════════════════════════

exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
all_images = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in exts)
already_done = {f.stem for f in OUTPUT_DIR.glob("*.glb")}
images = [f for f in all_images if safe_stem(f.name) not in already_done]

print(f"\n📂 Output: {OUTPUT_DIR}")
print(f"✅ {len(all_images)} image(s), {len(already_done)} already done, {len(images)} to process")

if not images:
    print("\n✅ All images already processed!")
    raise SystemExit()

for i, f in enumerate(images, 1):
    print(f"   {i}. {f.name} ({f.stat().st_size // 1024} KB)")

print(f"\n⚙  Texture: {CFG['texture_size']}px | Decimate: {CFG['decimate_target']:,} | Remesh: {CFG['remesh']}")

# ══════════════════════════════════════════════════════════════
# PROCESS EACH IMAGE — fully sequential, no parallelism
# ══════════════════════════════════════════════════════════════

MAX_RETRIES = 3
t_total = time.perf_counter()
results = []

for idx, img_path in enumerate(images):
    base = safe_stem(img_path.name)
    print(f"\n{'━' * 60}")
    print(f"[{idx+1}/{len(images)}] {img_path.name}")
    print(f"{'━' * 60}")

    t_img = time.perf_counter()
    error = None

    for attempt in range(MAX_RETRIES):
        try:
            # ── 1. Reconstruct ──
            cleanup()
            image = Image.open(img_path).convert("RGBA")

            if attempt > 0:
                print(f"  🔄 Attempt {attempt+1}/{MAX_RETRIES} — reloading models...")
                pipe.cuda()

            print(f"  ▸ Reconstructing...")
            t0 = time.perf_counter()
            out = pipe.run(
                [image], image_weights=[1.0],
                sparse_structure_sampler_params={"steps": CFG["ss_steps"]},
                shape_slat_sampler_params={"steps": CFG["shape_steps"]},
                tex_slat_sampler_params={"steps": CFG["tex_steps"]},
            )
            if not out:
                raise RuntimeError("Empty pipeline result")
            mesh = out[0]

            # Clone tensors — cumesh can lose storage references
            mesh.vertices = mesh.vertices.clone()
            mesh.faces = mesh.faces.clone()
            if hasattr(mesh, 'attrs') and mesh.attrs is not None:
                mesh.attrs = mesh.attrs.clone()
            if hasattr(mesh, 'coords') and mesh.coords is not None:
                mesh.coords = mesh.coords.clone()

            recon_s = round(time.perf_counter() - t0, 2)
            peak = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"  ✓ Recon: {recon_s}s | peak: {peak:.1f}GB | "
                  f"{mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")

            # ── 2. Render snapshot (optional, non-fatal) ──
            render_s = 0
            n_faces = mesh.faces.shape[0]
            if CFG["render_mode"] != "none" and n_faces <= RENDER_MAX_FACES:
                try:
                    t0 = time.perf_counter()
                    if CFG["render_mode"] == "video":
                        result = render_utils.render_video(
                            mesh, envmap=envmap,
                            resolution=CFG["video_resolution"],
                            num_frames=120,
                        )
                        frames = render_utils.make_pbr_vis_frames(
                            result, resolution=CFG["video_resolution"]
                        )
                        imageio.mimsave(str(OUTPUT_DIR / f"{base}.mp4"), frames, fps=15)
                        del frames, result
                    elif CFG["render_mode"] == "snapshot":
                        result = render_utils.render_video(
                            mesh, envmap=envmap,
                            resolution=CFG["video_resolution"],
                            num_frames=1,
                        )
                        frame = render_utils.make_pbr_vis_frames(
                            result, resolution=CFG["video_resolution"]
                        )[0]
                        Image.fromarray(frame).save(str(OUTPUT_DIR / f"{base}_preview.png"))
                        del frame, result
                    render_s = round(time.perf_counter() - t0, 2)
                    print(f"  ✓ Render: {render_s}s")
                except Exception as e:
                    print(f"  ⚠  Render failed (non-fatal): {e}")
                    # Check if CUDA context is poisoned (illegal memory access)
                    if not cuda_ok():
                        print(f"  ⚠  CUDA context corrupted by render — must retry from scratch")
                        raise RuntimeError("CUDA context corrupted after render failure")
            elif n_faces > RENDER_MAX_FACES:
                print(f"  ⚠  Skipping render ({n_faces:,} faces > {RENDER_MAX_FACES:,} limit)")

            # ── 3. Offload models → CPU ──
            del out
            safe_offload()
            print(f"  ✓ Models offloaded | {vram_free():.1f}GB free")

            # ── 4. Prepare mesh ──
            print(f"  ▸ Preparing mesh (remesh={CFG['remesh']})...")
            t0 = time.perf_counter()
            prepared = pp.prepare_mesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                voxel_size=mesh.voxel_size,
                decimation_target=CFG["decimate_target"],
                texture_size=CFG["texture_size"],
                remesh=CFG["remesh"],
                remesh_band=CFG.get("remesh_band", 1.0),
                verbose=True,
                name=base,
            )
            prepare_s = round(time.perf_counter() - t0, 2)
            print(f"  ✓ Prepare: {prepare_s}s")
            del mesh

            # ── 5. xatlas UV unwrap ──
            print(f"  ▸ UV unwrapping (xatlas)...")
            t0 = time.perf_counter()
            unwrapped = pp.uv_unwrap(prepared, verbose=True)
            xatlas_s = round(time.perf_counter() - t0, 2)
            print(f"  ✓ xatlas: {xatlas_s}s")
            del prepared

            # ── 6. Texture bake + GLB export ──
            glb_path = OUTPUT_DIR / f"{base}.glb"
            print(f"  ▸ Baking textures + exporting GLB...")
            t0 = time.perf_counter()
            pp.bake_and_export(unwrapped, str(glb_path), verbose=True)
            bake_s = round(time.perf_counter() - t0, 2)
            glb_size = glb_path.stat().st_size
            print(f"  ✓ Bake: {bake_s}s | GLB: {fmt_bytes(glb_size)}")
            del unwrapped

            # ── Done ──
            total_s = round(time.perf_counter() - t_img, 2)
            results.append(dict(
                name=base, recon=recon_s, render=render_s,
                prepare=prepare_s, xatlas=xatlas_s, bake=bake_s,
                total=total_s, size=glb_size, error=None,
            ))
            print(f"\n  ✅ {base} done in {total_s}s")
            error = None
            break  # success — exit retry loop

        except Exception as e:
            err = str(e).lower()
            retryable = ("storage" in err or "out of memory" in err
                         or "illegal memory" in err or "cuda error" in err
                         or "accelerator" in err)
            if attempt < MAX_RETRIES - 1 and retryable:
                print(f"  ⚠  Attempt {attempt+1} failed: {e}")
                print(f"  🔄 Cleaning up and retrying...")
                try: del out
                except: pass
                try: del mesh
                except: pass
                try: del prepared
                except: pass
                try: del unwrapped
                except: pass
                safe_offload()
                time.sleep(2)
            else:
                error = str(e)
                break

    if error:
        total_s = round(time.perf_counter() - t_img, 2)
        results.append(dict(name=base, total=total_s, error=error,
                            recon=0, render=0, prepare=0, xatlas=0, bake=0, size=0))
        print(f"\n  ❌ {base} failed: {error}")
        traceback.print_exc()

    # ── Full cleanup between images ──
    safe_offload()

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

wall = round(time.perf_counter() - t_total, 2)
ok = [r for r in results if not r["error"]]
fail = [r for r in results if r["error"]]

print(f"\n{'=' * 60}")
print("📊 RESULTS")
print(f"{'=' * 60}")
print(f"  Done: {len(ok)}/{len(results)}" + (f" ({len(fail)} failed)" if fail else ""))
print(f"  Wall: {wall}s ({wall/60:.1f}m)")

if ok:
    print(f"\n{'─' * 75}")
    print(f"  {'File':<35} {'Recon':>6} {'Prep':>6} {'xatlas':>7} {'Bake':>6} {'Total':>7}")
    print(f"  {'─' * 69}")
    for r in ok:
        n = r['name'][:32] + '..' if len(r['name']) > 34 else r['name']
        print(f"  {n:<35} {r['recon']:5.0f}s {r['prepare']:5.0f}s {r['xatlas']:6.0f}s {r['bake']:5.0f}s {r['total']:6.0f}s ✅")

for r in fail:
    print(f"  {r['name']:<35} {'':>6} {'':>6} {'':>7} {'':>6} {r['total']:6.0f}s ❌ {r['error'][:30]}")

print(f"\n📁 {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("*")):
    print(f"   {f.name}  ({fmt_bytes(f.stat().st_size)})")

print(f"\n✅ Done in {wall}s")