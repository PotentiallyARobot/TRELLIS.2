# ============================================================
# 🔺 TRELLIS.2 — 2D to 3D Generator + Background Removal
# Single cell: loads pipeline, launches custom web UI.
#
# Render modes:
#   - none:        Skip rendering entirely (fastest)
#   - snapshot:    Single-frame PNG (4 angles, PBR grid)
#   - video:       120-frame MP4 with bobbing camera + PBR grid
#   - perspective: Clean 360° turntable MP4 (shaded only, fixed pitch)
#
# Pipeline: postprocess_parallel (prepare → xatlas → bake)
# Safety: tensor cloning, face-count render guard, CUDA health checks
# ============================================================

import os, sys, pathlib, subprocess, re, time, threading, traceback, json, uuid, collections, shutil, gc

os.environ["TRELLIS2_DISABLE_REMBG"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask"])
    from flask import Flask, request, jsonify, send_file, Response

IN_COLAB = False
try:
    from google.colab.output import eval_js
    IN_COLAB = True
except ImportError:
    eval_js = None

# ── Console capture ──
class TeeWriter:
    def __init__(self, original, buf):
        self._original = original
        self._buf = buf
    def write(self, s):
        self._original.write(s)
        if s.strip():
            self._buf.append(s.rstrip('\n'))
        return len(s)
    def flush(self):
        self._original.flush()
    def __getattr__(self, name):
        return getattr(self._original, name)

console_lines = collections.deque(maxlen=800)
sys.stdout = TeeWriter(sys.__stdout__, console_lines)
sys.stderr = TeeWriter(sys.__stderr__, console_lines)

# ── Weight caching ──
DRIVE_WEIGHTS = pathlib.Path("/content/drive/MyDrive/trellis2_weights_local")
LOCAL_WEIGHTS = pathlib.Path("/content/trellis2_weights_local")
HF_MODEL_ID  = "microsoft/TRELLIS.2-4B"

def dir_size_bytes(p):
    total = 0
    for f in pathlib.Path(p).rglob("*"):
        if f.is_file(): total += f.stat().st_size
    return total

def copy_weights(src, dst, label=""):
    src = pathlib.Path(src); dst = pathlib.Path(dst)
    if dst.exists(): shutil.rmtree(dst)
    total = dir_size_bytes(src); copied = 0
    file_count = sum(1 for f in src.rglob("*") if f.is_file())
    print(f"  Copying {file_count} files ({total / 1e9:.1f} GB) {label}...")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src); target = dst / rel
        if item.is_dir(): target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))
            copied += item.stat().st_size
            pct = int(copied / total * 100) if total else 100
            sys.__stdout__.write(f"\r  {pct}% ({copied / 1e9:.1f} / {total / 1e9:.1f} GB)   ")
            sys.__stdout__.flush()
    sys.__stdout__.write("\n"); sys.__stdout__.flush()
    print(f"  ✅ Copy complete.")

def resolve_weights():
    if LOCAL_WEIGHTS.exists() and any(LOCAL_WEIGHTS.iterdir()):
        print(f"✅ Local weights found at {LOCAL_WEIGHTS}"); return str(LOCAL_WEIGHTS)
    if DRIVE_WEIGHTS.exists() and any(DRIVE_WEIGHTS.iterdir()):
        print(f"📂 Found cached weights on Drive: {DRIVE_WEIGHTS}")
        try: copy_weights(DRIVE_WEIGHTS, LOCAL_WEIGHTS, label="Drive → local"); return str(LOCAL_WEIGHTS)
        except Exception as e: print(f"  ⚠ Copy failed ({e}), downloading from HuggingFace.")
    print(f"⬇ Downloading weights from {HF_MODEL_ID}..."); return HF_MODEL_ID

def cache_weights_to_drive():
    if DRIVE_WEIGHTS.exists() and any(DRIVE_WEIGHTS.iterdir()): return
    src = LOCAL_WEIGHTS if LOCAL_WEIGHTS.exists() else None
    if not src:
        try:
            from huggingface_hub import snapshot_download
            src = pathlib.Path(snapshot_download(HF_MODEL_ID, local_files_only=True))
        except: print("  ⚠ Cannot find weights to cache."); return
    weight_size = dir_size_bytes(src)
    print(f"\n💾 Saving weights to Drive ({weight_size / 1e9:.1f} GB)...")
    try:
        usage = shutil.disk_usage("/content/drive/MyDrive")
        if usage.free / 1e9 < weight_size / 1e9 + 1.0:
            print(f"   ⚠ Not enough Drive space. Skipping."); return
    except: pass
    try: copy_weights(src, DRIVE_WEIGHTS, label="local → Drive")
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
        if DRIVE_WEIGHTS.exists():
            try: shutil.rmtree(DRIVE_WEIGHTS)
            except: pass

# ── Pipeline ──
REPO_DIR = pathlib.Path("/content/TRELLIS.2")
import torch, torch.nn as nn
import numpy as np
from PIL import Image
import cv2, imageio

try: torch.backends.cuda.matmul.fp32_precision = "tf32"
except: pass
try: torch.backends.cudnn.conv.fp32_precision = "tf32"
except: pass
torch.set_float32_matmul_precision("high")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import postprocess_parallel as pp

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9

# Max faces for render — above this the nvdiffrec renderer can trigger
# illegal memory access which poisons the entire CUDA context.
RENDER_MAX_FACES = 16_000_000

print(f"GPU: {GPU_NAME} | VRAM: {TOTAL_VRAM:.1f} GB")
print("Loading TRELLIS.2 pipeline...")
weights_path = resolve_weights()
downloaded_from_hf = (weights_path == HF_MODEL_ID)
trellis_pipe = Trellis2ImageTo3DPipeline.from_pretrained(weights_path)
trellis_pipe.cuda()

if downloaded_from_hf:
    try:
        from huggingface_hub import snapshot_download
        hf_cache_path = snapshot_download(HF_MODEL_ID, local_files_only=True)
        if not LOCAL_WEIGHTS.exists():
            print(f"\n📁 Copying HF cache to {LOCAL_WEIGHTS}...")
            copy_weights(hf_cache_path, LOCAL_WEIGHTS, label="HF cache → local")
    except Exception as e: print(f"  ⚠ Could not copy HF cache: {e}")
    threading.Thread(target=cache_weights_to_drive, daemon=True).start()

hdri = REPO_DIR / "assets" / "hdri" / "forest.exr"
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread(str(hdri), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device="cuda",
))
print("✅ TRELLIS.2 pipeline loaded")

# ── CUDA safety helpers ──
def cuda_ok():
    try:
        torch.cuda.synchronize()
        return True
    except:
        return False

def safe_cleanup():
    gc.collect()
    try: torch.cuda.empty_cache()
    except: pass
    gc.collect()

def safe_offload_models():
    """Offload ALL pipeline models to CPU. Logs any failures."""
    freed_before = torch.cuda.memory_allocated() / 1e9
    # Offload all named models
    for name, model in trellis_pipe.models.items():
        try:
            model.to("cpu")
        except Exception as e:
            print(f"    ⚠ offload {name} failed: {e}")
    # Offload image conditioning model
    try:
        trellis_pipe.image_cond_model.to("cpu")
    except Exception as e:
        print(f"    ⚠ offload image_cond_model failed: {e}")
    # Sweep any remaining nn.Module attributes on the pipeline
    for attr_name in dir(trellis_pipe):
        try:
            attr = getattr(trellis_pipe, attr_name)
            if isinstance(attr, torch.nn.Module) and any(
                p.is_cuda for p in attr.parameters()
            ):
                attr.to("cpu")
        except:
            pass
    safe_cleanup()
    freed_after = torch.cuda.memory_allocated() / 1e9
    freed = freed_before - freed_after
    print(f"    📤 Models offloaded: {freed:.1f}GB freed | {TOTAL_VRAM - freed_after:.1f}GB VRAM free")

def safe_reload_models():
    trellis_pipe.cuda()

# ── RMBG lazy ──
rmbg_pipe = None
rmbg_lock = threading.Lock()
def get_rmbg():
    global rmbg_pipe
    if rmbg_pipe is None:
        from transformers import pipeline as hf_pipeline
        if not hasattr(torch.nn.Module, "_patched_all_tied_weights_keys"):
            torch.nn.Module._patched_all_tied_weights_keys = True
            @property
            def _atwk(self): return {}
            setattr(torch.nn.Module, "all_tied_weights_keys", _atwk)
        rmbg_pipe = hf_pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    return rmbg_pipe

# ── State ──
UPLOAD_DIR = pathlib.Path("/content/_trellis_uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
jobs = {}; active_jobs = {}; _gpu_lock = threading.Lock()

def safe_stem(name):
    s = pathlib.Path(name).stem.strip(); s = re.sub(r"\s+","_",s); s = re.sub(r"[^A-Za-z0-9._-]+","",s)
    return s or "image"

def fmt_bytes(n):
    units=["B","KB","MB","GB","TB"]; f=float(max(n,0)); i=0
    while f>=1024.0 and i<len(units)-1: f/=1024.0; i+=1
    return f"{int(f)} {units[i]}" if i==0 else f"{f:.2f} {units[i]}"


# ══════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ══════════════════════════════════════════════════════════════

def do_render(mesh, mode, out_path, base, fps=15, resolution=1024):
    """
    Render preview using full PBR pipeline, one frame at a time.
    Each frame is moved to CPU immediately and GPU cache is freed.
    """
    n_faces = mesh.faces.shape[0]
    if mode == "none":
        return None, None
    if n_faces > RENDER_MAX_FACES:
        print(f"    ⚠  Skipping render ({n_faces:,} faces > {RENDER_MAX_FACES:,} limit)")
        return None, None

    try:
        # Camera path
        num_frames = 1 if mode == "snapshot" else 120
        yaws = (-torch.linspace(0, 2 * 3.1415, num_frames) + np.pi / 2).tolist()
        if mode == "perspective":
            pitch = [0.3] * num_frames
        else:
            pitch = (0.25 + 0.5 * torch.sin(
                torch.linspace(0, 2 * 3.1415, num_frames)
            )).tolist()
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitch, rs=2, fovs=40,
        )

        # Create renderer once
        renderer = render_utils.get_renderer(mesh, resolution=resolution)

        # Render ONE frame at a time, immediately move to CPU
        all_frames = {}
        for j in range(num_frames):
            res = renderer.render(mesh, extrinsics[j], intrinsics[j], envmap=envmap)
            for k, v in res.items():
                if k not in all_frames:
                    all_frames[k] = []
                if v.dim() == 2:
                    v = v[None].repeat(3, 1, 1)
                all_frames[k].append(
                    np.clip(v.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                )
            # Free ALL gpu tensors from this frame
            del res
            torch.cuda.empty_cache()

        del renderer
        torch.cuda.empty_cache()

        if mode == "snapshot":
            frame = render_utils.make_pbr_vis_frames(all_frames)[0]
            png_path = out_path / f"{base}_preview.png"
            Image.fromarray(frame).save(str(png_path))
            del frame, all_frames
            return str(png_path), "image"

        elif mode == "video":
            frames = render_utils.make_pbr_vis_frames(all_frames)
            mp4_path = out_path / f"{base}.mp4"
            imageio.mimsave(str(mp4_path), frames, fps=fps)
            del frames, all_frames
            return str(mp4_path), "video"

        elif mode == "perspective":
            frames = all_frames.get('shaded', [])
            mp4_path = out_path / f"{base}_perspective.mp4"
            imageio.mimsave(str(mp4_path), frames, fps=fps)
            del frames, all_frames
            return str(mp4_path), "video"

        else:
            del all_frames
            return None, None

    except Exception as e:
        print(f"    ⚠  Render failed ({mode}): {e}")
        if not cuda_ok():
            raise RuntimeError("CUDA context corrupted after render failure")
        return None, None


# ══════════════════════════════════════════════════════════════
# GENERATION JOB — fully sequential per image
# ══════════════════════════════════════════════════════════════

STEPS = [
    ("Loading image...",       0.01),
    ("Running 3D reconstruction...", 0.30),
    ("Preparing mesh...",      0.10),
    ("UV unwrapping (xatlas)...", 0.20),
    ("Baking textures + GLB...", 0.24),
    ("Rendering preview...",   0.15),
]

MAX_RETRIES = 3

def run_generate_job(job_id):
    job = jobs[job_id]; active_jobs["generate"] = job_id
    s = job["settings"]; files = job["files"]
    out_path = pathlib.Path(s["output_dir"]); out_path.mkdir(parents=True, exist_ok=True)
    total = len(files); done = 0; t0_all = time.perf_counter()

    with _gpu_lock:
        for idx, (orig_name, file_path) in enumerate(files):
            base = safe_stem(orig_name)
            glb_out = out_path / f"{base}.glb"

            def set_phase(si):
                label, _ = STEPS[si]
                cum = sum(w for _, w in STEPS[:si])
                pct = (idx + cum) / total
                job["progress"] = {
                    "pct": round(pct * 100, 1),
                    "image_num": idx + 1, "total": total,
                    "name": orig_name, "phase": label,
                    "elapsed": round(time.perf_counter() - t0_all, 1),
                }

            set_phase(0)
            job["log"].append(f"[{idx+1}/{total}] Processing: {orig_name}")
            t0 = time.perf_counter()

            error = None
            for attempt in range(MAX_RETRIES):
                try:
                    # ── 1. Reconstruct ──
                    # Aggressive cleanup to give cumesh the cleanest possible
                    # GPU state. The fill_holes() call inside pipe.run() can
                    # hit "Tensor without storage" if CUDA memory is fragmented.
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()

                    image = Image.open(file_path).convert("RGBA")

                    if attempt > 0:
                        job["log"].append(f"  🔄 Retry {attempt+1}/{MAX_RETRIES}...")
                        # Full pipeline reset on retry
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except: pass
                        safe_reload_models()

                    set_phase(1)
                    out = trellis_pipe.run(
                        [image], image_weights=[1.0],
                        sparse_structure_sampler_params={"steps": 12},
                        shape_slat_sampler_params={"steps": 12},
                        tex_slat_sampler_params={"steps": 12},
                    )
                    if not out:
                        raise RuntimeError("Empty pipeline result")
                    mesh = out[0]

                    # Clone tensors to prevent storage invalidation
                    mesh.vertices = mesh.vertices.clone()
                    mesh.faces = mesh.faces.clone()
                    if hasattr(mesh, 'attrs') and mesh.attrs is not None:
                        mesh.attrs = mesh.attrs.clone()
                    if hasattr(mesh, 'coords') and mesh.coords is not None:
                        mesh.coords = mesh.coords.clone()

                    recon_s = round(time.perf_counter() - t0, 2)
                    job["log"].append(
                        f"  ✓ Recon: {recon_s}s | "
                        f"{mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces"
                    )

                    # ── 2. Simplify for nvdiffrec render limit ──
                    # The working code does: mesh.simplify(16777216) then render.
                    # nvdiffrec cannot handle >16M faces.
                    n_raw = mesh.faces.shape[0]
                    if n_raw > 16777216:
                        job["log"].append(f"  ▸ Simplifying for render: {n_raw:,} → 16,777,216 faces")
                        mesh.simplify(16777216)

                    # ── 3. Render preview (BEFORE offload — matches working code) ──
                    # Models + mesh + envmap all on GPU, exactly like the reference.
                    set_phase(5)
                    render_mode = s.get("render_mode", "video")
                    media_path, media_type = None, None
                    if render_mode != "none":
                        free_gb = TOTAL_VRAM - torch.cuda.memory_allocated() / 1e9
                        job["log"].append(f"  ▸ Rendering ({render_mode}) | {free_gb:.1f}GB free")
                        media_path, media_type = do_render(
                            mesh, render_mode, out_path, base,
                            fps=s["fps"],
                        )
                        if media_path:
                            job["log"].append(f"  ✓ Render: {fmt_bytes(pathlib.Path(media_path).stat().st_size)}")
                        # Free render intermediates
                        torch.cuda.empty_cache()

                    # ── 4. Offload models → CPU for mesh processing ──
                    del out
                    safe_offload_models()

                    # ── 5. Prepare mesh (remesh/simplify/BVH) ──
                    set_phase(2)
                    t_prep = time.perf_counter()
                    prepared = pp.prepare_mesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        attr_volume=mesh.attrs,
                        coords=mesh.coords,
                        attr_layout=mesh.layout,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        voxel_size=mesh.voxel_size,
                        decimation_target=s["decimate_target"],
                        texture_size=s["texture_size"],
                        remesh=s["remesh"],
                        remesh_band=s["remesh_band"],
                        verbose=True,
                        name=base,
                    )
                    prep_s = round(time.perf_counter() - t_prep, 2)
                    job["log"].append(f"  ✓ Prepare: {prep_s}s")
                    del mesh
                    safe_cleanup()

                    # ── 6. xatlas UV unwrap ──
                    set_phase(3)
                    t_uv = time.perf_counter()
                    unwrapped = pp.uv_unwrap(prepared, verbose=True)
                    uv_s = round(time.perf_counter() - t_uv, 2)
                    job["log"].append(f"  ✓ xatlas: {uv_s}s")
                    del prepared

                    # ── 7. Texture bake + GLB export ──
                    set_phase(4)
                    t_bake = time.perf_counter()
                    pp.bake_and_export(unwrapped, str(glb_out), verbose=True)
                    bake_s = round(time.perf_counter() - t_bake, 2)
                    glb_size = glb_out.stat().st_size
                    job["log"].append(f"  ✓ Bake: {bake_s}s | GLB: {fmt_bytes(glb_size)}")
                    del unwrapped
                    safe_cleanup()

                    # ── Done ──
                    dt = round(time.perf_counter() - t0, 2)
                    result_entry = {"name": base, "glb": str(glb_out)}
                    if media_path:
                        result_entry["media"] = media_path
                        result_entry["media_type"] = media_type
                    job["log"].append(f"  ✅ {base} — GLB: {fmt_bytes(glb_size)} ({dt}s)")
                    job["results"].append(result_entry)
                    done += 1
                    error = None
                    break  # exit retry loop

                except Exception as e:
                    err = str(e).lower()
                    retryable = ("storage" in err or "out of memory" in err
                                 or "illegal memory" in err or "cuda error" in err
                                 or "accelerator" in err)
                    if attempt < MAX_RETRIES - 1 and retryable:
                        job["log"].append(f"  ⚠ Attempt {attempt+1} failed: {e}")
                        try: del out
                        except: pass
                        try: del mesh
                        except: pass
                        try: del prepared
                        except: pass
                        try: del unwrapped
                        except: pass
                        safe_offload_models()
                        gc.collect()
                        try: torch.cuda.synchronize()
                        except: pass
                        gc.collect()
                        try: torch.cuda.empty_cache()
                        except: pass
                        free = TOTAL_VRAM - torch.cuda.memory_allocated() / 1e9
                        job["log"].append(f"    Cleanup done | {free:.1f}GB free")
                        time.sleep(3)
                    else:
                        error = str(e)
                        break

            if error:
                job["log"].append(f"  ❌ {orig_name}: {error}")
                traceback.print_exc()

            # Cleanup between images
            safe_offload_models()

            # Reload for next image if not last
            if idx < total - 1:
                safe_reload_models()

        dt_total = time.perf_counter() - t0_all
        job["log"].append(f"\nDone — {done}/{total} in {dt_total:.1f}s")
        job["status"] = "done"
        job["progress"] = {
            "pct": 100, "image_num": total, "total": total,
            "name": "Complete", "phase": "All done!",
            "elapsed": round(dt_total, 1),
        }


def run_rmbg_job(job_id):
    job = jobs[job_id]; active_jobs["rmbg"] = job_id; files = job["files"]
    total = len(files); done = 0; t0 = time.perf_counter()
    with rmbg_lock:
        job["progress"] = {"pct": 0, "image_num": 0, "total": total,
                           "name": "Loading model...", "phase": "Loading RMBG-1.4...", "elapsed": 0}
        job["log"].append("Loading background removal model...")
        rmbg = get_rmbg()
        job["log"].append("Model loaded.")
        for idx, (orig_name, file_path) in enumerate(files):
            base = safe_stem(orig_name)
            out_p = pathlib.Path(file_path).parent / f"{base}_transparent.png"
            job["progress"] = {"pct": round((idx/total)*100, 1), "image_num": idx+1,
                               "total": total, "name": orig_name,
                               "phase": "Removing background...",
                               "elapsed": round(time.perf_counter()-t0, 1)}
            job["log"].append(f"[{idx+1}/{total}] {orig_name}")
            try:
                rgba = rmbg(str(file_path)); rgba.save(str(out_p), "PNG")
                job["log"].append(f"  ✅ {base}_transparent.png")
                job["results"].append({"name": base, "file": str(out_p), "original": orig_name})
                done += 1
            except Exception as e:
                job["log"].append(f"  ❌ {orig_name}: {e}")
                traceback.print_exc()
        dt = time.perf_counter() - t0
        job["log"].append(f"\nDone — {done}/{total} in {dt:.1f}s")
        job["status"] = "done"
        job["progress"] = {"pct": 100, "image_num": total, "total": total,
                           "name": "Complete", "phase": "All done!",
                           "elapsed": round(dt, 1)}


# ══════════════════════════════════════════════════════════════
# FLASK
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.route("/api/keepalive")
def api_keepalive():
    return jsonify({"ok": True})


# ── HTML PAGE ──

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRELLIS.2 — 2D to 3D Generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{--gold:#E8A917;--gold-light:#F5C842;--gold-dark:#C48E0E;--black:#0A0A0A;--off-black:#141414;--dark:#1A1A1A;--dark-border:#2A2A2A;--white:#F5F2EB;--gray:#8A8A8A;--gray-light:#B0B0B0;--red:#E84C4C;--green:#4CE870;--font-display:'Archivo Black',Impact,sans-serif;--font-mono:'JetBrains Mono',monospace;--font-body:'DM Sans',sans-serif}
body{background:var(--black);color:var(--white);font-family:var(--font-body);line-height:1.6;-webkit-font-smoothing:antialiased}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:9999}
a{color:var(--gold);text-decoration:none}.shell{max-width:960px;margin:0 auto;padding:24px}
.header{display:flex;align-items:center;gap:16px;padding-bottom:16px;border-bottom:1px solid var(--dark-border)}.header img{height:32px}.header-text h1{font-family:var(--font-display);font-size:1.15rem;letter-spacing:.5px}.header-text h1 span{color:var(--gold)}.header-text p{font-size:.78rem;color:var(--gray);margin-top:2px}
.banner{background:linear-gradient(135deg,rgba(232,169,23,.06),rgba(232,169,23,.02));border:1px solid rgba(232,169,23,.15);border-radius:10px;padding:16px 20px;margin:20px 0;font-size:.85rem;color:var(--gray-light);line-height:1.7}.banner strong{color:var(--white)}.banner .hl{color:var(--gold);font-weight:600}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--dark-border);margin-bottom:24px}.tab-btn{background:none;border:none;padding:14px 24px;font-family:var(--font-body);font-size:.92rem;font-weight:600;color:var(--gray);cursor:pointer;border-bottom:2px solid transparent;transition:color .2s,border-color .2s;position:relative;top:1px}.tab-btn:hover{color:var(--gray-light)}.tab-btn.active{color:var(--gold);border-bottom-color:var(--gold)}
.tab-panel{display:none}.tab-panel.active{display:block}
.instructions{background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;padding:16px 20px;margin-bottom:20px;font-size:.88rem;color:var(--gray-light);line-height:1.7}.instructions strong{color:var(--white)}.instructions .warn{color:var(--gold);font-weight:600}
.dropzone{border:2px dashed var(--dark-border);border-radius:12px;padding:48px 24px;text-align:center;cursor:pointer;position:relative;transition:border-color .2s,background .2s}.dropzone.over{border-color:var(--gold);background:rgba(232,169,23,.04)}.dropzone-icon{font-size:2.5rem;margin-bottom:12px;opacity:.5}.dropzone-text{color:var(--gray-light);font-size:.95rem}.dropzone-text strong{color:var(--gold)}.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.thumbs{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.thumb{position:relative;width:72px;height:72px;border-radius:8px;overflow:hidden;border:1px solid var(--dark-border)}.thumb img{width:100%;height:100%;object-fit:cover}.thumb-x{position:absolute;top:2px;right:2px;width:20px;height:20px;border-radius:50%;background:rgba(0,0,0,.7);color:var(--white);font-size:12px;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center}
.settings-toggle{background:none;border:1px solid var(--dark-border);color:var(--gray-light);font-family:var(--font-body);font-size:.88rem;font-weight:600;padding:10px 16px;border-radius:8px;cursor:pointer;margin-top:16px;width:100%;text-align:left;display:flex;justify-content:space-between;align-items:center;transition:border-color .2s}.settings-toggle:hover{border-color:var(--gray)}.settings-toggle .arrow{transition:transform .2s}.settings-toggle.open .arrow{transform:rotate(180deg)}
.settings-panel{display:none;margin-top:12px;background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;padding:20px}.settings-panel.open{display:block}
.field{margin-bottom:16px}.field:last-child{margin-bottom:0}.field label{display:block;font-size:.72rem;font-weight:600;color:var(--gray);margin-bottom:6px;font-family:var(--font-mono);letter-spacing:.5px;text-transform:uppercase}.field input,.field select{width:100%;background:var(--dark);border:1px solid var(--dark-border);border-radius:6px;padding:10px 12px;color:var(--white);font-family:var(--font-body);font-size:.9rem;outline:none;transition:border-color .2s}.field input:focus,.field select:focus{border-color:var(--gold)}.field .hint{font-size:.72rem;color:var(--gray);margin-top:4px}.field-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.checkbox-row{display:flex;align-items:center;gap:8px;font-size:.9rem;color:var(--gray-light);cursor:pointer}.checkbox-row input[type=checkbox]{width:18px;height:18px;accent-color:var(--gold);cursor:pointer}
/* Render mode selector */
.render-modes{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:6px}
.render-mode{background:var(--dark);border:1px solid var(--dark-border);border-radius:8px;padding:10px 8px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
.render-mode:hover{border-color:var(--gray)}
.render-mode.selected{border-color:var(--gold);background:rgba(232,169,23,.06)}
.render-mode .rm-icon{font-size:1.2rem;margin-bottom:4px}
.render-mode .rm-label{font-size:.78rem;font-weight:600;color:var(--white)}
.render-mode .rm-desc{font-size:.65rem;color:var(--gray);margin-top:2px;line-height:1.3}
.render-mode.selected .rm-label{color:var(--gold)}
.gen-btn{width:100%;margin-top:20px;padding:16px;background:var(--gold);color:var(--black);font-family:var(--font-body);font-weight:700;font-size:1rem;border:none;border-radius:8px;cursor:pointer;transition:background .2s,transform .15s;display:flex;align-items:center;justify-content:center;gap:8px}.gen-btn:hover:not(:disabled){background:var(--gold-light);transform:translateY(-1px)}.gen-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.progress-panel{margin-top:24px;display:none;background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;overflow:hidden}.progress-panel.active{display:block}
.progress-top{padding:16px 20px;border-bottom:1px solid var(--dark-border)}.progress-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.progress-status{display:flex;align-items:center;gap:8px;font-size:.9rem;font-weight:600}.progress-status .spinner{display:inline-block;width:16px;height:16px;border:2px solid var(--dark-border);border-top-color:var(--gold);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.progress-timer{font-family:var(--font-mono);font-size:.85rem;color:var(--gray)}.progress-phase{font-family:var(--font-mono);font-size:.8rem;color:var(--gold);margin-bottom:10px}
.progress-bar-track{width:100%;height:8px;background:var(--dark-border);border-radius:4px;overflow:hidden}.progress-bar-fill{height:100%;background:linear-gradient(90deg,var(--gold-dark),var(--gold),var(--gold-light));border-radius:4px;transition:width .6s ease;width:0%;position:relative}.progress-bar-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.15),transparent);animation:shimmer 1.5s infinite}@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}.progress-pct{text-align:right;font-family:var(--font-mono);font-size:.75rem;color:var(--gray);margin-top:6px}.progress-image{font-size:.82rem;color:var(--gray-light);margin-top:4px}
.console-section{border-top:1px solid var(--dark-border)}.console-header{padding:10px 20px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}.console-header:hover{background:rgba(255,255,255,.02)}.console-title{font-family:var(--font-mono);font-size:.75rem;color:var(--gray);display:flex;align-items:center;gap:8px}.console-title .dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}.console-arrow{color:var(--gray);font-size:.7rem;transition:transform .2s}.console-header.open .console-arrow{transform:rotate(180deg)}.console-body{max-height:0;overflow:hidden;transition:max-height .3s ease}.console-body.open{max-height:400px}.console-scroll{padding:12px 20px;font-family:var(--font-mono);font-size:.72rem;color:#6A9955;line-height:1.6;white-space:pre-wrap;max-height:280px;overflow-y:auto;background:#08080A}
.log-box{margin-top:12px;background:var(--off-black);border:1px solid var(--dark-border);border-radius:8px;padding:14px;font-family:var(--font-mono);font-size:.78rem;color:var(--gray-light);max-height:180px;overflow-y:auto;line-height:1.7;white-space:pre-wrap;display:none}.log-box.active{display:block}
.results{margin-top:28px;display:none}.results.active{display:block}.results-header{font-family:var(--font-display);font-size:1rem;margin-bottom:16px}
.result-card{background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;overflow:hidden;margin-bottom:16px;transition:border-color .2s}.result-card:hover{border-color:rgba(232,169,23,.3)}.result-card video,.result-card img.result-img{width:100%;display:block;background:var(--black)}.result-card img.result-img{max-height:300px;object-fit:contain;padding:12px;background:repeating-conic-gradient(#1a1a1a 0% 25%,#222 0% 50%) 0 0/20px 20px}.result-info{padding:14px 16px;display:flex;align-items:center;justify-content:space-between;gap:12px}.result-name{font-weight:600;font-size:.92rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.result-actions{display:flex;gap:8px;flex-shrink:0}.dl-btn{background:rgba(232,169,23,.1);border:1px solid rgba(232,169,23,.2);color:var(--gold);font-family:var(--font-mono);font-size:.75rem;font-weight:600;padding:6px 12px;border-radius:5px;cursor:pointer;transition:background .2s;text-decoration:none}.dl-btn:hover{background:rgba(232,169,23,.2);color:var(--gold-light)}
.no-preview{padding:40px 20px;text-align:center;color:var(--gray);font-size:.85rem}
.keepalive-badge{position:fixed;bottom:12px;right:12px;font-family:var(--font-mono);font-size:.65rem;color:var(--gray);opacity:.4}
@media(max-width:600px){.shell{padding:16px}.field-row{grid-template-columns:1fr}.result-info{flex-direction:column;align-items:flex-start}.tab-btn{padding:12px 14px;font-size:.82rem}.render-modes{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="shell">
  <div class="header"><img src="https://raw.githubusercontent.com/PotentiallyARobot/MissingLink/main/assets/logo.png" alt="MissingLink"><div class="header-text"><h1>2D to 3D <span>Generator</span></h1><p>Powered by TRELLIS.2 &amp; MissingLink</p></div></div>
  <div class="banner"><strong>MissingLink</strong> provides <span class="hl">ultra-fast, optimized 3D generation</span> on A100, L4, and T4 GPUs. Heavily optimized precompiled wheels eliminate 30+ minute build times — just pip install and run. Easy-to-use Colab notebooks get you generating in under 60 seconds.</div>
  <div class="tabs"><button class="tab-btn active" data-tab="generate" onclick="switchTab('generate',this)">🔺 Generate 3D</button><button class="tab-btn" data-tab="rmbg" onclick="switchTab('rmbg',this)">✂ Remove Background</button></div>

  <!-- ── GENERATE TAB ── -->
  <div class="tab-panel active" id="tab-generate">
    <div class="instructions"><strong>Upload your images and generate 3D models.</strong><br>Each image becomes a downloadable GLB model with optional preview render, saved to Google Drive.<br><br><span class="warn">⚠ Images must have transparent backgrounds (PNG with alpha).</span> Use the <strong>Remove Background</strong> tab first if needed.</div>
    <div class="dropzone" id="dropzone3d"><div class="dropzone-icon">📁</div><div class="dropzone-text">Drag &amp; drop images here, or <strong>click to browse</strong></div><input type="file" id="fileInput3d" multiple accept="image/*"></div>
    <div class="thumbs" id="thumbs3d"></div>
    <button class="settings-toggle" onclick="this.classList.toggle('open');document.getElementById('settingsPanel').classList.toggle('open')">⚙ Generation settings <span class="arrow">▾</span></button>
    <div class="settings-panel" id="settingsPanel">
      <div class="field"><label>Output directory</label><input type="text" id="sOutDir" value="/content/drive/MyDrive/trellis_models_out"><div class="hint">Google Drive path for GLB + preview output</div></div>

      <div class="field">
        <label>Render mode</label>
        <div class="render-modes" id="renderModes">
          <div class="render-mode" data-mode="none" onclick="selectRender(this)">
            <div class="rm-icon">⚡</div>
            <div class="rm-label">No render</div>
            <div class="rm-desc">GLB only, fastest</div>
          </div>
          <div class="render-mode" data-mode="snapshot" onclick="selectRender(this)">
            <div class="rm-icon">📷</div>
            <div class="rm-label">Snapshot</div>
            <div class="rm-desc">Single preview image</div>
          </div>
          <div class="render-mode selected" data-mode="video" onclick="selectRender(this)">
            <div class="rm-icon">🎬</div>
            <div class="rm-label">Video</div>
            <div class="rm-desc">360° orbit preview</div>
          </div>
          <div class="render-mode" data-mode="perspective" onclick="selectRender(this)">
            <div class="rm-icon">🔄</div>
            <div class="rm-label">Perspective</div>
            <div class="rm-desc">Clean 360° turntable</div>
          </div>
        </div>
      </div>

      <div class="field-row"><div class="field"><label>Video FPS</label><input type="number" id="sFps" value="15" min="5" max="60"></div><div class="field"><label>Texture size</label><select id="sTexture"><option>1024</option><option>2048</option><option selected>4096</option><option>8192</option></select></div></div>
      <div class="field-row"><div class="field"><label>Max faces</label><input type="number" id="sDecimate" value="1000000" min="1000"><div class="hint">Target face count for render + exported GLB</div></div><div class="field"><label>Remesh band</label><input type="number" id="sRemeshBand" value="1.0" min="0.1" max="3.0" step="0.1"></div></div>
      <label class="checkbox-row"><input type="checkbox" id="sRemesh" checked> Enable remeshing</label>
    </div>
    <button class="gen-btn" id="genBtn3d" onclick="startGen()" disabled>Generate 3D models →</button>
    <div class="progress-panel" id="progressPanel3d"><div class="progress-top"><div class="progress-row"><div class="progress-status" id="pStatus3d"><span class="spinner"></span> Generating...</div><div class="progress-timer" id="pTimer3d">0:00</div></div><div class="progress-phase" id="pPhase3d">Starting...</div><div class="progress-bar-track"><div class="progress-bar-fill" id="pFill3d"></div></div><div class="progress-pct" id="pPct3d">0%</div><div class="progress-image" id="pImage3d"></div></div><div class="console-section"><div class="console-header open" id="consoleHead3d" onclick="toggleConsole()"><div class="console-title"><span class="dot"></span> Console output</div><span class="console-arrow">▾</span></div><div class="console-body open" id="consoleBody3d"><div class="console-scroll" id="consoleScroll3d"></div></div></div></div>
    <div class="log-box" id="logBox3d"></div>
    <div class="results" id="results3d"><div class="results-header">✅ Results</div><div id="resultsList3d"></div></div>
  </div>

  <!-- ── RMBG TAB ── -->
  <div class="tab-panel" id="tab-rmbg">
    <div class="instructions"><strong>Remove backgrounds from your images before 3D generation.</strong><br>Upload images with backgrounds — processed through RMBG-1.4 into transparent PNGs. Model loads on first use.</div>
    <div class="dropzone" id="dropzoneRmbg"><div class="dropzone-icon">✂</div><div class="dropzone-text">Drag &amp; drop images here, or <strong>click to browse</strong></div><input type="file" id="fileInputRmbg" multiple accept="image/*"></div>
    <div class="thumbs" id="thumbsRmbg"></div>
    <button class="gen-btn" id="genBtnRmbg" onclick="startRmbg()" disabled>Remove backgrounds →</button>
    <div class="progress-panel" id="progressPanelRmbg"><div class="progress-top"><div class="progress-row"><div class="progress-status" id="pStatusRmbg"><span class="spinner"></span> Processing...</div><div class="progress-timer" id="pTimerRmbg">0:00</div></div><div class="progress-phase" id="pPhaseRmbg">Starting...</div><div class="progress-bar-track"><div class="progress-bar-fill" id="pFillRmbg"></div></div><div class="progress-pct" id="pPctRmbg">0%</div></div></div>
    <div class="log-box" id="logBoxRmbg"></div>
    <div class="results" id="resultsRmbg"><div class="results-header">✅ Backgrounds removed</div><div id="resultsListRmbg"></div></div>
  </div>
</div>
<div class="keepalive-badge" id="keepaliveBadge">●</div>
<script>
function $(id){return document.getElementById(id)}function enc(s){return encodeURIComponent(s)}function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}function show(id){$(id).classList.add('active')}function hide(id){$(id).classList.remove('active')}function fmtTime(s){const m=Math.floor(s/60),sec=Math.floor(s%60);return m+':'+String(sec).padStart(2,'0')}
function switchTab(name,el){document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));el.classList.add('active');$('tab-'+name).classList.add('active')}
function toggleConsole(){$('consoleHead3d').classList.toggle('open');$('consoleBody3d').classList.toggle('open')}

/* ── Render mode selector ── */
let selectedRenderMode='video';
function selectRender(el){document.querySelectorAll('.render-mode').forEach(m=>m.classList.remove('selected'));el.classList.add('selected');selectedRenderMode=el.dataset.mode}

/* ── Dropzone ── */
function initDrop(zId,iId,tId,bId,arr){const z=$(zId),inp=$(iId);['dragenter','dragover'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.add('over')}));['dragleave','drop'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.remove('over')}));z.addEventListener('drop',ev=>addF(ev.dataTransfer.files,arr,tId,bId));inp.addEventListener('change',ev=>{addF(ev.target.files,arr,tId,bId);inp.value=''})}
function addF(fl,arr,tId,bId){for(const f of fl)if(f.type.startsWith('image/'))arr.push(f);renderT(arr,tId,bId)}
function renderT(arr,tId,bId){const t=$(tId);t.innerHTML='';arr.forEach((f,i)=>{const d=document.createElement('div');d.className='thumb';const img=document.createElement('img');img.src=URL.createObjectURL(f);const x=document.createElement('button');x.className='thumb-x';x.textContent='×';x.onclick=()=>{arr.splice(i,1);renderT(arr,tId,bId)};d.append(img,x);t.append(d)});$(bId).disabled=arr.length===0}
let files3d=[],filesRmbg=[];initDrop('dropzone3d','fileInput3d','thumbs3d','genBtn3d',files3d);initDrop('dropzoneRmbg','fileInputRmbg','thumbsRmbg','genBtnRmbg',filesRmbg);

/* ── Keepalive ── */
setInterval(async()=>{try{const r=await fetch('/api/keepalive');$('keepaliveBadge').style.color=r.ok?'var(--green)':'var(--red)'}catch(e){$('keepaliveBadge').style.color='var(--red)'}setTimeout(()=>{$('keepaliveBadge').style.color='var(--gray)'},2000)},60000);

/* ── Polling ── */
let timers={},localStart={};
function poll(jobId,type,cfg){if(timers[type])clearInterval(timers[type]);if(!localStart[type])localStart[type]=Date.now();timers[type]=setInterval(async()=>{try{const r=await fetch('/api/status/'+jobId);const d=await r.json();const p=d.progress||{};const elapsed=p.elapsed||((Date.now()-localStart[type])/1000);$(cfg.timer).textContent=fmtTime(elapsed);const pct=p.pct||0;$(cfg.fill).style.width=pct+'%';$(cfg.pct).textContent=Math.round(pct)+'%';$(cfg.phase).textContent=p.phase||'';if(cfg.image&&p.image_num&&p.total)$(cfg.image).textContent='Image '+p.image_num+' of '+p.total+(p.name?' — '+p.name:'');if(d.status==='done')$(cfg.status).innerHTML='<span class="done-icon">✅</span> Complete';else $(cfg.status).innerHTML='<span class="spinner"></span> '+cfg.statusText;if(d.log){const b=$(cfg.log);b.textContent=d.log.join('\n');b.scrollTop=b.scrollHeight}if(cfg.console){const cr=await fetch('/api/console');const cd=await cr.json();const cel=$(cfg.console);cel.textContent=cd.lines.join('\n');cel.scrollTop=cel.scrollHeight}if(d.status==='done'){clearInterval(timers[type]);timers[type]=null;delete localStart[type];if(cfg.btn){cfg.btn.disabled=false;cfg.btn.textContent=cfg.btnText}cfg.renderFn(d.results||[])}}catch(e){console.error(e)}},800)}

/* ── Generate ── */
async function startGen(){if(!files3d.length)return;const btn=$('genBtn3d');btn.disabled=true;btn.textContent='Uploading images...';const fd=new FormData();files3d.forEach(f=>fd.append('images',f));fd.append('settings',JSON.stringify({output_dir:$('sOutDir').value,fps:parseInt($('sFps').value),texture_size:parseInt($('sTexture').value),decimate_target:parseInt($('sDecimate').value),remesh:$('sRemesh').checked,remesh_band:parseFloat($('sRemeshBand').value),render_mode:selectedRenderMode,video_resolution:512}));try{const r=await fetch('/api/generate',{method:'POST',body:fd});const d=await r.json();if(!d.job_id)throw new Error(d.error||'Failed');btn.textContent='Generating...';show('progressPanel3d');show('logBox3d');$('logBox3d').textContent='';hide('results3d');$('resultsList3d').innerHTML='';$('pFill3d').style.width='0%';$('pPct3d').textContent='0%';localStart['generate']=Date.now();poll(d.job_id,'generate',{timer:'pTimer3d',fill:'pFill3d',pct:'pPct3d',phase:'pPhase3d',status:'pStatus3d',statusText:'Generating...',image:'pImage3d',log:'logBox3d',console:'consoleScroll3d',btn:btn,btnText:'Generate 3D models →',renderFn:render3d})}catch(e){alert('Error: '+e.message);btn.disabled=false;btn.textContent='Generate 3D models →'}}

/* ── Render results — handles all media types ── */
function render3d(results){if(!results.length)return;show('results3d');const l=$('resultsList3d');l.innerHTML='';results.forEach(r=>{const c=document.createElement('div');c.className='result-card';let mediaHtml='';if(r.media&&r.media_type==='video'){mediaHtml=`<video src="/api/file?p=${enc(r.media)}" controls playsinline autoplay muted loop></video>`}else if(r.media&&r.media_type==='image'){mediaHtml=`<img class="result-img" src="/api/file?p=${enc(r.media)}" alt="${esc(r.name)}">`}else{mediaHtml='<div class="no-preview">No preview — GLB only</div>'}let btns=`<a class="dl-btn" href="/api/file?p=${enc(r.glb)}" download="${esc(r.name)}.glb">GLB</a>`;if(r.media){const ext=r.media_type==='video'?'mp4':'png';btns=`<a class="dl-btn" href="/api/file?p=${enc(r.media)}" download="${esc(r.name)}.${ext}">${ext.toUpperCase()}</a>`+btns}c.innerHTML=mediaHtml+`<div class="result-info"><div class="result-name">${esc(r.name)}</div><div class="result-actions">${btns}</div></div>`;l.append(c)})}

/* ── RMBG ── */
async function startRmbg(){if(!filesRmbg.length)return;const btn=$('genBtnRmbg');btn.disabled=true;btn.textContent='Uploading...';const fd=new FormData();filesRmbg.forEach(f=>fd.append('images',f));try{const r=await fetch('/api/rmbg',{method:'POST',body:fd});const d=await r.json();if(!d.job_id)throw new Error(d.error||'Failed');btn.textContent='Processing...';show('progressPanelRmbg');show('logBoxRmbg');$('logBoxRmbg').textContent='';hide('resultsRmbg');$('resultsListRmbg').innerHTML='';$('pFillRmbg').style.width='0%';$('pPctRmbg').textContent='0%';localStart['rmbg']=Date.now();poll(d.job_id,'rmbg',{timer:'pTimerRmbg',fill:'pFillRmbg',pct:'pPctRmbg',phase:'pPhaseRmbg',status:'pStatusRmbg',statusText:'Processing...',image:null,log:'logBoxRmbg',console:null,btn:btn,btnText:'Remove backgrounds →',renderFn:renderRmbg})}catch(e){alert('Error: '+e.message);btn.disabled=false;btn.textContent='Remove backgrounds →'}}
function renderRmbg(results){if(!results.length)return;show('resultsRmbg');const l=$('resultsListRmbg');l.innerHTML='';results.forEach(r=>{const c=document.createElement('div');c.className='result-card';c.innerHTML=`<img class="result-img" src="/api/file?p=${enc(r.file)}" alt="${esc(r.name)}"><div class="result-info"><div class="result-name">${esc(r.name)}_transparent.png</div><div class="result-actions"><a class="dl-btn" href="/api/file?p=${enc(r.file)}" download="${esc(r.name)}_transparent.png">Download PNG</a></div></div>`;l.append(c)})}

/* ── Reconnect ── */
async function tryReconnect(){try{const r=await fetch('/api/active');const d=await r.json();if(d.generate){show('progressPanel3d');show('logBox3d');$('genBtn3d').disabled=true;$('genBtn3d').textContent='Generating...';localStart['generate']=Date.now();poll(d.generate,'generate',{timer:'pTimer3d',fill:'pFill3d',pct:'pPct3d',phase:'pPhase3d',status:'pStatus3d',statusText:'Generating...',image:'pImage3d',log:'logBox3d',console:'consoleScroll3d',btn:$('genBtn3d'),btnText:'Generate 3D models →',renderFn:render3d})}if(d.rmbg){switchTab('rmbg',document.querySelector('[data-tab=rmbg]'));show('progressPanelRmbg');show('logBoxRmbg');$('genBtnRmbg').disabled=true;$('genBtnRmbg').textContent='Processing...';localStart['rmbg']=Date.now();poll(d.rmbg,'rmbg',{timer:'pTimerRmbg',fill:'pFillRmbg',pct:'pPctRmbg',phase:'pPhaseRmbg',status:'pStatusRmbg',statusText:'Processing...',image:null,log:'logBoxRmbg',console:null,btn:$('genBtnRmbg'),btnText:'Remove backgrounds →',renderFn:renderRmbg})}}catch(e){}}
tryReconnect();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images"}), 400
    settings = json.loads(request.form.get("settings", "{}"))
    for k, v in [("output_dir", "/content/drive/MyDrive/trellis_models_out"),
                 ("fps", 15), ("texture_size", 4096), ("decimate_target", 1000000),
                 ("remesh", True), ("remesh_band", 1.0), ("render_mode", "video"),
                 ("video_resolution", 512)]:
        settings.setdefault(k, v)
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id; job_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = job_dir / f.filename; f.save(str(dest))
        saved.append((f.filename, str(dest)))
    jobs[job_id] = {
        "status": "running",
        "progress": {"pct": 0, "image_num": 0, "total": len(saved),
                     "name": "Starting...", "phase": "Preparing...", "elapsed": 0},
        "log": [], "results": [], "files": saved, "settings": settings,
    }
    threading.Thread(target=run_generate_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/rmbg", methods=["POST"])
def api_rmbg():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images"}), 400
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id; job_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = job_dir / f.filename; f.save(str(dest))
        saved.append((f.filename, str(dest)))
    jobs[job_id] = {
        "status": "running",
        "progress": {"pct": 0, "image_num": 0, "total": len(saved),
                     "name": "Starting...", "phase": "Loading model...", "elapsed": 0},
        "log": [], "results": [], "files": saved, "settings": {},
    }
    threading.Thread(target=run_rmbg_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"],
                    "log": job["log"], "results": job["results"]})

@app.route("/api/console")
def api_console():
    return jsonify({"lines": list(console_lines)[-200:]})

@app.route("/api/active")
def api_active():
    r = {}
    for k in ["generate", "rmbg"]:
        j = active_jobs.get(k)
        if j and j in jobs and jobs[j]["status"] == "running":
            r[k] = j
    return jsonify(r)

@app.route("/api/file")
def api_file():
    p = request.args.get("p", "")
    if not p or not os.path.isfile(p):
        return "Not found", 404
    return send_file(p)


# ══════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════

def run_server():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)

public_url = None

if IN_COLAB:
    try:
        public_url = eval_js("google.colab.kernel.proxyPort(5000, {'cache': false})")
        if public_url and not public_url.startswith("http"):
            public_url = "https://" + public_url
    except Exception as e:
        sys.__stdout__.write(f"⚠ eval_js proxyPort failed: {e}\n")

    if not public_url:
        try:
            hostname = eval_js("window.location.hostname")
            if hostname:
                kernel_url = eval_js("window.location.href")
                sys.__stdout__.write(f"  Kernel URL: {kernel_url}\n")
        except Exception as e:
            sys.__stdout__.write(f"⚠ hostname fallback failed: {e}\n")

    if public_url:
        try:
            from IPython.display import display, HTML
            display(HTML(f'''
            <div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;">
                <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🔺 TRELLIS.2 Generator is live — click to open:</div>
                <a href="{public_url}" target="_blank" style="color:#E8A917;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
            </div>
            '''))
        except:
            print(f"\n🔺 TRELLIS.2 Generator is live:\n   {public_url}\n")
    else:
        print("\n⚠ Could not get Colab proxy URL.")
        print("  Try opening this manually in your browser:")
        print("  Look at your Colab URL and replace 'colab.research.google.com' with")
        print("  '5000-{your-vm-id}.{region}.prod.colab.dev'\n")
        print("  Or run this in a new cell:")
        print("  from google.colab.output import eval_js")
        print("  print(eval_js('google.colab.kernel.proxyPort(5000)'))\n")
else:
    print("\n🔺 TRELLIS.2 Generator running at http://localhost:5000\n")

print("Server running. Interrupt cell to stop.\n")
try:
    while True:
        time.sleep(30)
        sys.__stdout__.write(f"\r🔺 Uptime: {time.strftime('%H:%M:%S', time.gmtime(time.time()))} | Jobs: {len(jobs)}   ")
        sys.__stdout__.flush()
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped.")