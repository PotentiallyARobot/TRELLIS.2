# ============================================================
# 🔺 TRELLIS.2 — Single Image Processor (Safe Sequential Mode)
#
# Upload 1 or more images. Each is fully processed before the
# next one starts. No threads, no parallelism, no VRAM contention.
# ============================================================
import os, sys, time, re, gc, traceback, pathlib, shutil

import torch
import numpy as np
from PIL import Image
import cv2, imageio

sys.path.append("/content/TRELLIS.2")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import postprocess_parallel as pp

os.environ["TRELLIS2_DISABLE_REMBG"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_DIR = pathlib.Path("/content/TRELLIS.2")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

torch.set_float32_matmul_precision("high")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_stem(name):
    s = pathlib.Path(name).stem.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "image"


def _fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    gc.collect()


def _vram_free(total_vram):
    return total_vram - torch.cuda.memory_allocated() / 1e9


def _cuda_ok():
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def _safe_offload(pipe):
    for _, model in pipe.models.items():
        try:
            model.to("cpu")
        except RuntimeError:
            pass
    try:
        pipe.image_cond_model.to("cpu")
    except RuntimeError:
        pass
    _cleanup()


# ══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════

def run_trellis2_batch(
    # ── Input / Output ──
    input_dir: str = "/content/images_in",
    output_dir: str = "/content/drive/MyDrive/trellis_batch_output",
    upload_images: bool = True,

    # ── Pipeline ──
    model_name: str = "microsoft/TRELLIS.2-4B",
    low_vram: bool = True,
    hdri_path: str | None = None,

    # ── Sampler steps ──
    ss_steps: int = 12,
    shape_steps: int = 12,
    tex_steps: int = 12,

    # ── Mesh / Texture ──
    texture_size: int = 4096,
    decimate_target: int = 1_000_000,
    remesh: bool = True,
    remesh_band: float = 1.0,

    # ── Render ──
    render_mode: str = "snapshot",      # "video" | "snapshot" | "none"
    video_resolution: int = 512,
    video_num_frames: int = 120,
    video_fps: int = 15,
    render_max_faces: int = 16_000_000,

    # ── Retry ──
    max_retries: int = 3,
    retry_delay: float = 2.0,

    # ── Misc ──
    skip_already_done: bool = True,
    verbose: bool = True,
):
    """
    Process one or more images through the TRELLIS.2 pipeline, producing
    textured GLB meshes.

    Parameters
    ----------
    input_dir : str
        Directory containing input images (png/jpg/jpeg/webp/bmp).
    output_dir : str
        Directory where GLB files (and optional previews) are saved.
    upload_images : bool
        If True, trigger a Colab file-upload dialog to populate input_dir.
    model_name : str
        HuggingFace model identifier for the Trellis2 pipeline.
    low_vram : bool
        Enable low-VRAM mode (offloads models between stages).
    hdri_path : str | None
        Path to an HDRI .exr for the environment map. Defaults to the
        bundled forest.exr.
    ss_steps : int
        Sparse-structure sampler steps.
    shape_steps : int
        Shape SLAT sampler steps.
    tex_steps : int
        Texture SLAT sampler steps.
    texture_size : int
        Baked texture resolution in pixels (e.g. 2048, 4096).
    decimate_target : int
        Target face count after decimation.
    remesh : bool
        Whether to apply remeshing before UV unwrap.
    remesh_band : float
        Remesh band width parameter.
    render_mode : str
        "video" → full turntable MP4, "snapshot" → single preview PNG,
        "none" → skip rendering entirely.
    video_resolution : int
        Resolution for rendered video/snapshot frames.
    video_num_frames : int
        Number of frames for turntable video.
    video_fps : int
        FPS for turntable video.
    render_max_faces : int
        Skip rendering if face count exceeds this (avoids CUDA crashes).
    max_retries : int
        Number of attempts per image before marking as failed.
    retry_delay : float
        Seconds to wait between retries.
    skip_already_done : bool
        If True, skip images whose GLB already exists in output_dir.
    verbose : bool
        Print detailed progress information.

    Returns
    -------
    list[dict]
        One dict per image with keys: name, recon, render, prepare,
        xatlas, bake, total, size, error.
    """

    # ── GPU info ──
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    if verbose:
        print("=" * 60)
        print(f" GPU: {gpu_name} | VRAM: {total_vram:.1f} GB")
        print("=" * 60)

    # ── Upload (Colab) ──
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)

    if upload_images:
        from google.colab import files as colab_files
        print("📂 Upload your image(s):")
        uploaded = colab_files.upload()
        if not uploaded:
            raise SystemExit("No files uploaded.")
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        for name, data in uploaded.items():
            clean = re.sub(r'\s*\(\d+\)(?=\.\w+$)', '', pathlib.Path(name).name)
            (input_dir / clean).write_bytes(data)
        if verbose:
            print(f"✅ {len(uploaded)} image(s) uploaded")
    else:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ──
    if verbose:
        print("\n🔧 Loading pipeline...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_name)
    pipe.low_vram = low_vram
    pipe.cuda()

    hdri = pathlib.Path(hdri_path) if hdri_path else REPO_DIR / "assets" / "hdri" / "forest.exr"
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(str(hdri), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device="cuda",
    ))
    if verbose:
        print(f"✅ Pipeline loaded | alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Gather images ──
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in exts)

    if skip_already_done:
        already_done = {f.stem for f in output_dir.glob("*.glb")}
        images = [f for f in all_images if _safe_stem(f.name) not in already_done]
    else:
        already_done = set()
        images = list(all_images)

    if verbose:
        print(f"\n📂 Output: {output_dir}")
        print(f"✅ {len(all_images)} image(s), {len(already_done)} already done, {len(images)} to process")
        print(f"⚙  Texture: {texture_size}px | Decimate: {decimate_target:,} | Remesh: {remesh}")
        for i, f in enumerate(images, 1):
            print(f"   {i}. {f.name} ({f.stat().st_size // 1024} KB)")

    if not images:
        print("\n✅ All images already processed!")
        return []

    # ── Process each image ──
    t_total = time.perf_counter()
    results = []

    for idx, img_path in enumerate(images):
        base = _safe_stem(img_path.name)
        if verbose:
            print(f"\n{'━' * 60}")
            print(f"[{idx+1}/{len(images)}] {img_path.name}")
            print(f"{'━' * 60}")

        t_img = time.perf_counter()
        error = None

        for attempt in range(max_retries):
            try:
                # ── 1. Reconstruct ──
                _cleanup()
                image = Image.open(img_path).convert("RGBA")

                if attempt > 0:
                    if verbose:
                        print(f"  🔄 Attempt {attempt+1}/{max_retries} — reloading models...")
                    pipe.cuda()

                if verbose:
                    print(f"  ▸ Reconstructing...")
                t0 = time.perf_counter()
                out = pipe.run(
                    [image], image_weights=[1.0],
                    sparse_structure_sampler_params={"steps": ss_steps},
                    shape_slat_sampler_params={"steps": shape_steps},
                    tex_slat_sampler_params={"steps": tex_steps},
                )
                if not out:
                    raise RuntimeError("Empty pipeline result")
                mesh = out[0]

                mesh.vertices = mesh.vertices.clone()
                mesh.faces = mesh.faces.clone()
                if hasattr(mesh, 'attrs') and mesh.attrs is not None:
                    mesh.attrs = mesh.attrs.clone()
                if hasattr(mesh, 'coords') and mesh.coords is not None:
                    mesh.coords = mesh.coords.clone()

                recon_s = round(time.perf_counter() - t0, 2)
                peak = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                if verbose:
                    print(f"  ✓ Recon: {recon_s}s | peak: {peak:.1f}GB | "
                          f"{mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")

                # ── 2. Render (optional, non-fatal) ──
                render_s = 0
                n_faces = mesh.faces.shape[0]
                if render_mode != "none" and n_faces <= render_max_faces:
                    try:
                        t0 = time.perf_counter()
                        if render_mode == "video":
                            result = render_utils.render_video(
                                mesh, envmap=envmap,
                                resolution=video_resolution,
                                num_frames=video_num_frames,
                            )
                            frames = render_utils.make_pbr_vis_frames(
                                result, resolution=video_resolution
                            )
                            imageio.mimsave(str(output_dir / f"{base}.mp4"), frames, fps=video_fps)
                            del frames, result
                        elif render_mode == "snapshot":
                            result = render_utils.render_video(
                                mesh, envmap=envmap,
                                resolution=video_resolution,
                                num_frames=1,
                            )
                            frame = render_utils.make_pbr_vis_frames(
                                result, resolution=video_resolution
                            )[0]
                            Image.fromarray(frame).save(str(output_dir / f"{base}_preview.png"))
                            del frame, result
                        render_s = round(time.perf_counter() - t0, 2)
                        if verbose:
                            print(f"  ✓ Render: {render_s}s")
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠  Render failed (non-fatal): {e}")
                        if not _cuda_ok():
                            if verbose:
                                print(f"  ⚠  CUDA context corrupted — must retry")
                            raise RuntimeError("CUDA context corrupted after render failure")
                elif n_faces > render_max_faces and verbose:
                    print(f"  ⚠  Skipping render ({n_faces:,} faces > {render_max_faces:,} limit)")

                # ── 3. Offload models → CPU ──
                del out
                _safe_offload(pipe)
                if verbose:
                    print(f"  ✓ Models offloaded | {_vram_free(total_vram):.1f}GB free")

                # ── 4. Prepare mesh ──
                if verbose:
                    print(f"  ▸ Preparing mesh (remesh={remesh})...")
                t0 = time.perf_counter()
                prepared = pp.prepare_mesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    voxel_size=mesh.voxel_size,
                    decimation_target=decimate_target,
                    texture_size=texture_size,
                    remesh=remesh,
                    remesh_band=remesh_band,
                    verbose=verbose,
                    name=base,
                )
                prepare_s = round(time.perf_counter() - t0, 2)
                if verbose:
                    print(f"  ✓ Prepare: {prepare_s}s")
                del mesh

                # ── 5. xatlas UV unwrap ──
                if verbose:
                    print(f"  ▸ UV unwrapping (xatlas)...")
                t0 = time.perf_counter()
                unwrapped = pp.uv_unwrap(prepared, verbose=verbose)
                xatlas_s = round(time.perf_counter() - t0, 2)
                if verbose:
                    print(f"  ✓ xatlas: {xatlas_s}s")
                del prepared

                # ── 6. Texture bake + GLB export ──
                glb_path = output_dir / f"{base}.glb"
                if verbose:
                    print(f"  ▸ Baking textures + exporting GLB...")
                t0 = time.perf_counter()
                pp.bake_and_export(unwrapped, str(glb_path), verbose=verbose)
                bake_s = round(time.perf_counter() - t0, 2)
                glb_size = glb_path.stat().st_size
                if verbose:
                    print(f"  ✓ Bake: {bake_s}s | GLB: {_fmt_bytes(glb_size)}")
                del unwrapped

                # ── Done ──
                total_s = round(time.perf_counter() - t_img, 2)
                results.append(dict(
                    name=base, recon=recon_s, render=render_s,
                    prepare=prepare_s, xatlas=xatlas_s, bake=bake_s,
                    total=total_s, size=glb_size, error=None,
                ))
                if verbose:
                    print(f"\n  ✅ {base} done in {total_s}s")
                error = None
                break

            except Exception as e:
                err = str(e).lower()
                retryable = ("storage" in err or "out of memory" in err
                             or "illegal memory" in err or "cuda error" in err
                             or "accelerator" in err)
                if attempt < max_retries - 1 and retryable:
                    if verbose:
                        print(f"  ⚠  Attempt {attempt+1} failed: {e}")
                        print(f"  🔄 Cleaning up and retrying...")
                    for var in ('out', 'mesh', 'prepared', 'unwrapped'):
                        try: del locals()[var]
                        except: pass
                    _safe_offload(pipe)
                    time.sleep(retry_delay)
                else:
                    error = str(e)
                    break

        if error:
            total_s = round(time.perf_counter() - t_img, 2)
            results.append(dict(name=base, total=total_s, error=error,
                                recon=0, render=0, prepare=0, xatlas=0, bake=0, size=0))
            if verbose:
                print(f"\n  ❌ {base} failed: {error}")
                traceback.print_exc()

        _safe_offload(pipe)

    # ── Results summary ──
    wall = round(time.perf_counter() - t_total, 2)
    ok = [r for r in results if not r["error"]]
    fail = [r for r in results if r["error"]]

    if verbose:
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
        print(f"\n📁 {output_dir}")
        for f in sorted(output_dir.glob("*")):
            print(f"   {f.name}  ({_fmt_bytes(f.stat().st_size)})")
        print(f"\n✅ Done in {wall}s")

    return results


# ══════════════════════════════════════════════════════════════
# SAMPLE CALLS
# ══════════════════════════════════════════════════════════════

# # ── Max quality (default) ──
# results = run_trellis2_batch(
#     input_dir="/content/images_in",
#     output_dir="/content/drive/MyDrive/trellis_batch_output",
#     upload_images=True,
#     model_name="microsoft/TRELLIS.2-4B",
#     low_vram=True,
#     hdri_path=None,                   # uses bundled forest.exr
#     ss_steps=12,
#     shape_steps=12,
#     tex_steps=12,
#     texture_size=4096,
#     decimate_target=1_000_000,
#     remesh=True,
#     remesh_band=1.0,
#     render_mode="snapshot",           # "video" | "snapshot" | "none"
#     video_resolution=512,
#     video_num_frames=120,
#     video_fps=15,
#     render_max_faces=16_000_000,
#     max_retries=3,
#     retry_delay=2.0,
#     skip_already_done=True,
#     verbose=True,
# )

# # ── Fast preset equivalent ──
# results = run_trellis2_batch(
#     ss_steps=8, shape_steps=8, tex_steps=8,
#     texture_size=2048, decimate_target=500_000,
#     remesh=False, render_mode="snapshot",
# )