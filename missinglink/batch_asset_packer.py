# ============================================================
# 🔺 TRELLIS.2 — Batch Image Packer + Background Removal
# Single cell: loads RMBG-1.4, launches web UI for combining
# multiple asset images into a single packed image for bulk
# 3D processing with TRELLIS.2.
#
# Upload individual images → auto-remove backgrounds via
# RMBG-1.4 → trim & normalize → bin-pack into one combined
# PNG on a black canvas → download & feed into TRELLIS.2.
# ============================================================

def run_batch_packer():
    import os, sys, pathlib, subprocess, re, time, threading, traceback
    import json, uuid, collections, shutil, gc, math, io, base64

    # ── Flask ──
    try:
        from flask import Flask, request, jsonify, send_file, Response
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask"])
        from flask import Flask, request, jsonify, send_file, Response
    from werkzeug.utils import secure_filename

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

    console_lines = collections.deque(maxlen=400)
    sys.stdout = TeeWriter(sys.__stdout__, console_lines)
    sys.stderr = TeeWriter(sys.__stderr__, console_lines)

    # ── PIL / numpy ──
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "Pillow", "numpy"])
        from PIL import Image
        import numpy as np

    # ── RMBG-1.4 lazy loader ──
    rmbg_pipe = None
    rmbg_lock = threading.Lock()

    def _ensure_monkey_patch():
        import torch
        if not hasattr(torch.nn.Module, "_patched_all_tied_weights_keys"):
            torch.nn.Module._patched_all_tied_weights_keys = True
            @property
            def _atwk(self): return {}
            setattr(torch.nn.Module, "all_tied_weights_keys", _atwk)
            print("  🔧 Applied all_tied_weights_keys monkey patch for RMBG-1.4")

    def get_rmbg():
        nonlocal rmbg_pipe
        if rmbg_pipe is not None:
            return rmbg_pipe
        with rmbg_lock:
            if rmbg_pipe is not None:
                return rmbg_pipe
            print("🔄 Loading RMBG-1.4 background removal model...")
            t0 = time.perf_counter()
            try:
                _ensure_monkey_patch()
                from transformers import pipeline as hf_pipeline
                rmbg_pipe = hf_pipeline(
                    "image-segmentation",
                    model="briaai/RMBG-1.4",
                    trust_remote_code=True,
                    device=-1,  # CPU
                )
                dt = round(time.perf_counter() - t0, 1)
                print(f"✅ RMBG-1.4 loaded in {dt}s")
            except Exception as e:
                print(f"❌ RMBG-1.4 load failed: {e}")
                traceback.print_exc()
                raise
        return rmbg_pipe

    def has_transparency(img):
        if img.mode != "RGBA":
            return False
        alpha = np.array(img.getchannel("A"))
        non_opaque = np.sum(alpha < 250)
        ratio = non_opaque / alpha.size if alpha.size > 0 else 0
        return bool(ratio > 0.005)

    def remove_bg(image_path):
        """Remove background via RMBG-1.4, return RGBA PIL Image."""
        rmbg = get_rmbg()
        result = rmbg(str(image_path))
        return result  # PIL RGBA image

    def trim_transparent(img):
        """Crop to bounding box of non-transparent pixels, with small padding."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bbox = img.getbbox()
        if not bbox:
            return img
        pad = 4
        l = max(0, bbox[0] - pad)
        t = max(0, bbox[1] - pad)
        r = min(img.width, bbox[2] + pad)
        b = min(img.height, bbox[3] + pad)
        return img.crop((l, t, r, b))

    def normalize_size(img, max_dim):
        """Scale image so longest edge <= max_dim."""
        if max(img.width, img.height) <= max_dim:
            return img
        scale = max_dim / max(img.width, img.height)
        nw = max(1, round(img.width * scale))
        nh = max(1, round(img.height * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    def pack_images(sizes, padding=16):
        """
        Square-grid bin packing. Chooses the column count closest to
        sqrt(n) that produces the most square overall canvas, then
        lays images out in a uniform grid with each cell sized to the
        largest asset. Images are centered within their cell.
        Returns (placements, total_w, total_h).
        """
        n = len(sizes)
        if n == 0:
            return [], 0, 0

        # Target a square grid: cols ≈ ceil(sqrt(n))
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Uniform cell size = largest width/height across all images
        cell_w = max(w for w, h in sizes) + padding
        cell_h = max(h for w, h in sizes) + padding

        total_w = cols * cell_w + padding
        total_h = rows * cell_h + padding

        placements = []
        for i, (w, h) in enumerate(sizes):
            c = i % cols
            r = i // cols
            # Cell top-left
            cx = padding // 2 + c * cell_w
            cy = padding // 2 + r * cell_h
            # Center image within cell
            x = cx + (cell_w - w) // 2
            y = cy + (cell_h - h) // 2
            placements.append((x, y, w, h))

        return placements, total_w, total_h

    # ── State ──
    UPLOAD_DIR = pathlib.Path("/tmp/_batch_packer_uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR = pathlib.Path("/tmp/_batch_packer_output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    jobs = {}
    active_job = [None]

    def safe_stem(name):
        s = pathlib.Path(name).stem.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        return s or "image"

    def run_pack_job(job_id):
        job = jobs[job_id]
        active_job[0] = job_id
        files = job["files"]
        settings = job["settings"]
        total = len(files)
        t0 = time.perf_counter()
        max_dim = settings.get("max_size", 512)
        padding = settings.get("padding", 16)
        auto_rmbg = settings.get("auto_rmbg", True)

        processed_images = []
        processed_info = []

        for idx, (orig_name, file_path) in enumerate(files):
            base = safe_stem(orig_name)
            job["progress"] = {
                "pct": round((idx / total) * 100, 1),
                "current": idx + 1, "total": total,
                "name": orig_name,
                "phase": "Processing...",
                "elapsed": round(time.perf_counter() - t0, 1),
            }

            try:
                img = Image.open(file_path).convert("RGBA")
                already_transparent = has_transparency(img)

                if auto_rmbg and not already_transparent:
                    job["log"].append(f"[{idx+1}/{total}] {orig_name} — removing background...")
                    job["progress"]["phase"] = "Removing background..."
                    try:
                        img = remove_bg(file_path)
                        if img.mode != "RGBA":
                            img = img.convert("RGBA")
                        job["log"].append(f"  ✅ Background removed")
                    except Exception as e:
                        job["log"].append(f"  ⚠ RMBG failed: {e} — using original")
                        img = Image.open(file_path).convert("RGBA")
                else:
                    status = "has alpha" if already_transparent else "skipped (rmbg off)"
                    job["log"].append(f"[{idx+1}/{total}] {orig_name} — {status}")

                # Trim & normalize
                job["progress"]["phase"] = "Trimming..."
                trimmed = trim_transparent(img)
                normalized = normalize_size(trimmed, max_dim)

                # Save processed thumbnail for UI
                thumb_path = pathlib.Path(file_path).parent / f"{base}_processed.png"
                normalized.save(str(thumb_path), "PNG")

                processed_images.append(normalized)
                processed_info.append({
                    "name": orig_name,
                    "base": base,
                    "original_size": f"{img.width}×{img.height}",
                    "packed_size": f"{normalized.width}×{normalized.height}",
                    "thumb": str(thumb_path),
                    "had_alpha": already_transparent,
                })

                del img, trimmed
                gc.collect()

            except Exception as e:
                job["log"].append(f"[{idx+1}/{total}] ❌ {orig_name}: {e}")
                traceback.print_exc()

        if not processed_images:
            job["status"] = "error"
            job["log"].append("❌ No images processed successfully")
            return

        # ── Pack ──
        job["progress"]["phase"] = "Packing layout..."
        sizes = [(img.width, img.height) for img in processed_images]
        placements, total_w, total_h = pack_images(sizes, padding)

        # ── Composite onto black canvas ──
        job["progress"]["phase"] = "Compositing..."
        canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 255))
        for i, (x, y, w, h) in enumerate(placements):
            canvas.paste(processed_images[i], (x, y), processed_images[i])

        # Save result
        out_name = f"batch_packed_{job_id}.png"
        out_path = OUTPUT_DIR / out_name
        canvas.save(str(out_path), "PNG")

        used_area = sum(w * h for w, h in sizes)
        total_area = total_w * total_h
        utilization = round((used_area / total_area) * 100) if total_area > 0 else 0

        dt = time.perf_counter() - t0
        job["log"].append(f"\n✅ Packed {len(processed_images)} images → {total_w}×{total_h} ({utilization}% utilization) in {dt:.1f}s")
        job["status"] = "done"
        job["result"] = {
            "file": str(out_path),
            "filename": out_name,
            "width": total_w,
            "height": total_h,
            "count": len(processed_images),
            "utilization": utilization,
            "images": processed_info,
        }
        job["progress"] = {
            "pct": 100, "current": total, "total": total,
            "name": "Complete", "phase": "All done!",
            "elapsed": round(dt, 1),
        }

        del processed_images, canvas
        gc.collect()

    # ══════════════════════════════════════════════════════════════
    # FLASK
    # ══════════════════════════════════════════════════════════════
    app = Flask(__name__)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route("/api/keepalive")
    def api_keepalive():
        return jsonify({"ok": True})

    @app.route("/api/pack", methods=["POST"])
    def api_pack():
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images"}), 400
        settings = json.loads(request.form.get("settings", "{}"))
        for k, v in [("max_size", 512), ("padding", 16), ("auto_rmbg", True)]:
            settings.setdefault(k, v)

        job_id = uuid.uuid4().hex[:12]
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            safe_name = secure_filename(f.filename) or f"upload_{uuid.uuid4().hex[:8]}.png"
            dest = job_dir / safe_name
            f.save(str(dest))
            saved.append((f.filename, str(dest)))

        jobs[job_id] = {
            "status": "running",
            "progress": {"pct": 0, "current": 0, "total": len(saved),
                         "name": "Starting...", "phase": "Preparing...", "elapsed": 0},
            "log": [], "result": None, "files": saved, "settings": settings,
        }
        threading.Thread(target=run_pack_job, args=(job_id,), daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/api/status/<job_id>")
    def api_status(job_id):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Not found"}), 404
        # Safe serialization — convert numpy types to native Python
        payload = {
            "status": job["status"],
            "progress": job["progress"],
            "log": job["log"],
            "result": job["result"],
        }
        return Response(
            json.dumps(payload, default=_json_safe) + "\n",
            mimetype="application/json",
        )

    def _json_safe(obj):
        """Fallback serializer for numpy scalars and other non-standard types."""
        import numbers
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, numbers.Integral):
            return int(obj)
        if isinstance(obj, numbers.Real):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    @app.route("/api/file")
    def api_file():
        p = request.args.get("p", "")
        if not p:
            return "Not found", 404
        try:
            real = os.path.realpath(p)
        except (ValueError, OSError):
            return "Invalid path", 400
        if not os.path.isfile(real):
            return "Not found", 404
        # Only serve from our directories
        allowed = [os.path.realpath(str(UPLOAD_DIR)), os.path.realpath(str(OUTPUT_DIR))]
        if not any(real.startswith(a + os.sep) or real == a for a in allowed):
            return "Access denied", 403
        return send_file(real)

    @app.route("/api/console")
    def api_console():
        return jsonify({"lines": list(console_lines)[-200:]})

    # ── HTML PAGE ──
    HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRELLIS.2 — Batch Image Packer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{--gold:#E8A917;--gold-light:#F5C842;--gold-dark:#C48E0E;--black:#0A0A0A;--off-black:#141414;--dark:#1A1A1A;--dark-border:#2A2A2A;--white:#F5F2EB;--gray:#8A8A8A;--gray-light:#B0B0B0;--red:#E84C4C;--green:#4CE870;--blue:#4C9BE8;--font-display:'Archivo Black',Impact,sans-serif;--font-mono:'JetBrains Mono',monospace;--font-body:'DM Sans',sans-serif}
body{background:var(--black);color:var(--white);font-family:var(--font-body);line-height:1.6;-webkit-font-smoothing:antialiased}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:9999}
a{color:var(--gold);text-decoration:none}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--dark-border);border-radius:3px}
.shell{max-width:960px;margin:0 auto;padding:24px}
.header{display:flex;align-items:center;gap:16px;padding-bottom:16px;border-bottom:1px solid var(--dark-border)}
.header-tri{width:32px;height:32px;background:var(--gold);clip-path:polygon(50% 0%,6% 100%,94% 100%);filter:drop-shadow(0 0 10px rgba(232,169,23,.3))}
.header-text h1{font-family:var(--font-display);font-size:1.15rem;letter-spacing:.5px}.header-text h1 span{color:var(--gold)}.header-text p{font-size:.78rem;color:var(--gray);margin-top:2px}
.instructions{background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;padding:16px 20px;margin:20px 0;font-size:.88rem;color:var(--gray-light);line-height:1.7}.instructions strong{color:var(--white)}.instructions .hl{color:var(--gold);font-weight:600}

/* Dropzone */
.dropzone{border:2px dashed var(--dark-border);border-radius:12px;padding:48px 24px;text-align:center;cursor:pointer;position:relative;transition:border-color .2s,background .2s}
.dropzone.over{border-color:var(--gold);background:rgba(232,169,23,.04)}
.dropzone-icon{font-size:2.5rem;margin-bottom:12px;opacity:.5}
.dropzone-text{color:var(--gray-light);font-size:.95rem}.dropzone-text strong{color:var(--gold)}
.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer}

/* Thumbnails */
.thumbs{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
.thumb{position:relative;width:80px;height:80px;border-radius:8px;overflow:hidden;border:1px solid var(--dark-border);background:#000}
.thumb img{width:100%;height:100%;object-fit:contain;padding:4px}
.thumb-x{position:absolute;top:2px;right:2px;width:20px;height:20px;border-radius:50%;background:rgba(0,0,0,.8);color:var(--white);font-size:12px;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}
.thumb-x:hover{background:var(--red)}
.thumb-label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.75);font-size:.55rem;color:var(--gray-light);padding:2px 4px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:var(--font-mono)}

/* Settings */
.settings-card{background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;padding:20px;margin-top:16px}
.settings-title{font-family:var(--font-mono);font-size:.7rem;font-weight:600;color:var(--gray);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:14px}
.field{margin-bottom:14px}.field:last-child{margin-bottom:0}
.field label{display:flex;justify-content:space-between;align-items:center;font-size:.78rem;font-weight:600;color:var(--gray-light);margin-bottom:6px;font-family:var(--font-mono);letter-spacing:.3px}
.field label .val{color:var(--gold);font-weight:600}
.field input[type=range]{width:100%;accent-color:var(--gold);height:4px;background:var(--dark-border);border-radius:2px;outline:none;-webkit-appearance:none}
.field input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--gold);border-radius:50%;cursor:pointer;box-shadow:0 0 8px rgba(232,169,23,.3)}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.toggle-group{display:flex;gap:4px}
.toggle-btn{flex:1;padding:8px 0;border-radius:6px;border:1px solid var(--dark-border);background:var(--dark);color:var(--gray);font-family:var(--font-mono);font-size:.8rem;font-weight:600;cursor:pointer;transition:all .15s}
.toggle-btn.active{border-color:var(--gold);background:rgba(232,169,23,.08);color:var(--gold)}

/* Pack button */
.pack-btn{width:100%;margin-top:20px;padding:16px;background:var(--gold);color:var(--black);font-family:var(--font-body);font-weight:700;font-size:1rem;border:none;border-radius:8px;cursor:pointer;transition:background .2s,transform .15s;display:flex;align-items:center;justify-content:center;gap:8px}
.pack-btn:hover:not(:disabled){background:var(--gold-light);transform:translateY(-1px)}
.pack-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* Progress */
.progress-panel{margin-top:20px;display:none;background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;overflow:hidden}
.progress-panel.active{display:block}
.progress-top{padding:16px 20px}
.progress-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.progress-status{display:flex;align-items:center;gap:8px;font-size:.9rem;font-weight:600}
.progress-status .spinner{display:inline-block;width:16px;height:16px;border:2px solid var(--dark-border);border-top-color:var(--gold);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-timer{font-family:var(--font-mono);font-size:.85rem;color:var(--gray)}
.progress-phase{font-family:var(--font-mono);font-size:.8rem;color:var(--gold);margin-bottom:10px}
.progress-bar-track{width:100%;height:8px;background:var(--dark-border);border-radius:4px;overflow:hidden}
.progress-bar-fill{height:100%;background:linear-gradient(90deg,var(--gold-dark),var(--gold),var(--gold-light));border-radius:4px;transition:width .6s ease;width:0%;position:relative}
.progress-bar-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.15),transparent);animation:shimmer 1.5s infinite}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.progress-pct{text-align:right;font-family:var(--font-mono);font-size:.75rem;color:var(--gray);margin-top:6px}

/* Console */
.console-section{border-top:1px solid var(--dark-border)}
.console-header{padding:10px 20px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}
.console-header:hover{background:rgba(255,255,255,.02)}
.console-title{font-family:var(--font-mono);font-size:.75rem;color:var(--gray);display:flex;align-items:center;gap:8px}
.console-title .dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.console-arrow{color:var(--gray);font-size:.7rem;transition:transform .2s}
.console-header.open .console-arrow{transform:rotate(180deg)}
.console-body{max-height:0;overflow:hidden;transition:max-height .3s ease}
.console-body.open{max-height:400px}
.console-scroll{padding:12px 20px;font-family:var(--font-mono);font-size:.72rem;color:#6A9955;line-height:1.6;white-space:pre-wrap;max-height:280px;overflow-y:auto;background:#08080A}

/* Result */
.result-area{margin-top:24px;display:none}
.result-area.active{display:block}
.stats-bar{display:flex;gap:0;margin-bottom:16px;background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;overflow:hidden}
.stat{flex:1;padding:14px 16px;text-align:center;border-right:1px solid var(--dark-border)}
.stat:last-child{border-right:none}
.stat-label{font-family:var(--font-mono);font-size:.65rem;color:var(--gray);font-weight:600;letter-spacing:1px;text-transform:uppercase}
.stat-value{font-family:var(--font-mono);font-size:1.1rem;color:var(--gold);font-weight:700;margin-top:4px}
.preview-card{background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;overflow:hidden}
.preview-card img{width:100%;display:block;background:repeating-conic-gradient(#1a1a1a 0% 25%,#111 0% 50%) 0 0/20px 20px;max-height:600px;object-fit:contain;padding:12px}
.result-actions{padding:14px 16px;display:flex;gap:10px;border-top:1px solid var(--dark-border)}
.dl-btn{flex:1;padding:12px;text-align:center;border-radius:6px;font-family:var(--font-mono);font-size:.85rem;font-weight:700;cursor:pointer;transition:all .2s;text-decoration:none}
.dl-btn.primary{background:var(--gold);color:var(--black);border:none}
.dl-btn.primary:hover{background:var(--gold-light)}
.dl-btn.secondary{background:none;border:2px solid var(--dark-border);color:var(--gray-light)}
.dl-btn.secondary:hover{border-color:var(--gold);color:var(--gold)}

/* Image list in result */
.packed-list{margin-top:16px;background:var(--off-black);border:1px solid var(--dark-border);border-radius:10px;padding:16px}
.packed-list-title{font-family:var(--font-mono);font-size:.7rem;color:var(--gray);font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}
.packed-item{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--dark-border)}
.packed-item:last-child{border-bottom:none}
.packed-item img{width:40px;height:40px;object-fit:contain;border-radius:4px;background:#000;border:1px solid var(--dark-border)}
.packed-item .pi-name{font-size:.82rem;font-weight:500;color:var(--white);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.packed-item .pi-size{font-family:var(--font-mono);font-size:.7rem;color:var(--gray)}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-family:var(--font-mono);font-size:.6rem;font-weight:700;letter-spacing:.3px;text-transform:uppercase;margin-left:8px}
.badge-green{background:rgba(76,232,112,.1);color:var(--green)}
.badge-blue{background:rgba(76,155,232,.1);color:var(--blue)}

@media(max-width:600px){.shell{padding:16px}.field-row{grid-template-columns:1fr}.stats-bar{flex-wrap:wrap}.stat{min-width:50%}}
</style>
</head>
<body>
<div class="shell">
  <div class="header">
    <div class="header-tri"></div>
    <div class="header-text">
      <h1>Batch Image <span>Packer</span></h1>
      <p>Combine multiple assets into one image for TRELLIS.2 bulk 3D processing</p>
    </div>
  </div>

  <div class="instructions">
    <strong>Upload your individual asset images</strong> and pack them into a single combined image.<br>
    Backgrounds are <span class="hl">automatically removed via RMBG-1.4</span> (same model as the main TRELLIS.2 pipeline).
    Each image is trimmed, normalized, and bin-packed to maximize canvas utilization.
    Download the result and feed it into TRELLIS.2 for bulk 3D generation.
  </div>

  <div class="dropzone" id="dropzone">
    <div class="dropzone-icon">📂</div>
    <div class="dropzone-text">Drag &amp; drop images here, or <strong>click to browse</strong></div>
    <input type="file" id="fileInput" multiple accept="image/*">
  </div>
  <div class="thumbs" id="thumbs"></div>

  <div class="settings-card">
    <div class="settings-title">Settings</div>
    <div class="field">
      <label>Background Removal (RMBG-1.4)</label>
      <div class="toggle-group">
        <button class="toggle-btn active" data-val="on" onclick="setRmbg(true,this)">✂ Auto Remove</button>
        <button class="toggle-btn" data-val="off" onclick="setRmbg(false,this)">None</button>
      </div>
    </div>
    <div class="field-row">
      <div class="field">
        <label>Max Asset Size <span class="val" id="sizeVal">512px</span></label>
        <input type="range" min="128" max="1024" step="64" value="512" id="sizeSlider"
          oninput="maxSize=+this.value;document.getElementById('sizeVal').textContent=this.value+'px'">
      </div>
      <div class="field">
        <label>Padding <span class="val" id="padVal">16px</span></label>
        <input type="range" min="0" max="64" step="4" value="16" id="padSlider"
          oninput="padding=+this.value;document.getElementById('padVal').textContent=this.value+'px'">
      </div>
    </div>
  </div>

  <button class="pack-btn" id="packBtn" onclick="startPack()" disabled>
    Pack Images →
  </button>

  <div class="progress-panel" id="progressPanel">
    <div class="progress-top">
      <div class="progress-row">
        <div class="progress-status" id="pStatus"><span class="spinner"></span> Packing...</div>
        <div class="progress-timer" id="pTimer">0:00</div>
      </div>
      <div class="progress-phase" id="pPhase">Starting...</div>
      <div class="progress-bar-track"><div class="progress-bar-fill" id="pFill"></div></div>
      <div class="progress-pct" id="pPct">0%</div>
    </div>
    <div class="console-section">
      <div class="console-header open" id="consoleHead" onclick="toggleConsole()">
        <div class="console-title"><span class="dot"></span> Console output</div>
        <span class="console-arrow">▾</span>
      </div>
      <div class="console-body open" id="consoleBody">
        <div class="console-scroll" id="consoleScroll"></div>
      </div>
    </div>
  </div>

  <div class="result-area" id="resultArea">
    <div class="stats-bar" id="statsBar"></div>
    <div class="preview-card" id="previewCard"></div>
    <div class="packed-list" id="packedList"></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const enc=encodeURIComponent;

let uploadedFiles=[];
let maxSize=512, padding=16, autoRmbg=true;
let pollTimer=null, localStart=0;

/* ── Dropzone ── */
const dz=$('dropzone');
const fi=$('fileInput');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');addFiles(e.dataTransfer.files)});
fi.addEventListener('change',e=>{addFiles(e.target.files);fi.value=''});

function addFiles(fileList){
  const valid=['image/png','image/jpeg','image/webp','image/bmp','image/gif'];
  for(const f of fileList){
    if(!valid.includes(f.type))continue;
    uploadedFiles.push(f);
  }
  updateThumbs();
}

function removeFile(idx){
  uploadedFiles.splice(idx,1);
  updateThumbs();
  $('resultArea').classList.remove('active');
}

function updateThumbs(){
  const t=$('thumbs');
  t.innerHTML='';
  uploadedFiles.forEach((f,i)=>{
    const url=URL.createObjectURL(f);
    const d=document.createElement('div');d.className='thumb';
    d.innerHTML=`<img src="${url}" alt=""><button class="thumb-x" onclick="removeFile(${i})">×</button><div class="thumb-label">${esc(f.name)}</div>`;
    t.appendChild(d);
  });
  const btn=$('packBtn');
  btn.disabled=!uploadedFiles.length;
  btn.textContent=uploadedFiles.length?`Pack ${uploadedFiles.length} Image${uploadedFiles.length>1?'s':''} →`:'Pack Images →';
}

function setRmbg(val,btn){
  autoRmbg=val;
  document.querySelectorAll('.toggle-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}

/* ── Pack ── */
async function startPack(){
  if(!uploadedFiles.length)return;
  const btn=$('packBtn');
  btn.disabled=true;btn.textContent='Uploading...';

  const fd=new FormData();
  uploadedFiles.forEach(f=>fd.append('images',f));
  fd.append('settings',JSON.stringify({max_size:maxSize,padding:padding,auto_rmbg:autoRmbg}));

  try{
    const r=await fetch('/api/pack',{method:'POST',body:fd});
    const d=await r.json();
    if(!d.job_id)throw new Error(d.error||'Failed');

    btn.textContent='Packing...';
    $('progressPanel').classList.add('active');
    $('resultArea').classList.remove('active');
    $('pFill').style.width='0%';$('pPct').textContent='0%';
    $('consoleScroll').textContent='';
    localStart=Date.now();
    pollJob(d.job_id);
  }catch(e){
    alert('Error: '+e.message);
    btn.disabled=false;btn.textContent=`Pack ${uploadedFiles.length} Images →`;
  }
}

function pollJob(jobId){
  if(pollTimer)clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    try{
      // Update timer
      const elapsed=Math.floor((Date.now()-localStart)/1000);
      const m=Math.floor(elapsed/60),s=elapsed%60;
      $('pTimer').textContent=m+':'+(s<10?'0':'')+s;

      const r=await fetch(`/api/status/${jobId}`);
      const d=await r.json();
      if(d.error){clearInterval(pollTimer);return}

      const p=d.progress;
      $('pPhase').textContent=`[${p.current||0}/${p.total||0}] ${p.phase||''}`;
      $('pFill').style.width=p.pct+'%';
      $('pPct').textContent=Math.round(p.pct)+'%';

      // Log
      if(d.log&&d.log.length){
        $('consoleScroll').textContent=d.log.join('\n');
        const cs=$('consoleScroll');cs.scrollTop=cs.scrollHeight;
      }

      if(d.status==='done'){
        clearInterval(pollTimer);
        $('pStatus').innerHTML='✅ Complete';
        $('pFill').style.width='100%';$('pPct').textContent='100%';
        $('packBtn').disabled=false;
        $('packBtn').textContent=`Pack ${uploadedFiles.length} Image${uploadedFiles.length>1?'s':''} →`;
        if(d.result)showResult(d.result);
      }else if(d.status==='error'){
        clearInterval(pollTimer);
        $('pStatus').innerHTML='❌ Error';
        $('packBtn').disabled=false;
        $('packBtn').textContent=`Pack ${uploadedFiles.length} Images →`;
      }
    }catch(e){console.error('Poll error:',e)}
  },800);
}

function showResult(res){
  $('resultArea').classList.add('active');

  // Stats
  $('statsBar').innerHTML=`
    <div class="stat"><div class="stat-label">Assets</div><div class="stat-value">${res.count}</div></div>
    <div class="stat"><div class="stat-label">Dimensions</div><div class="stat-value">${res.width} × ${res.height}</div></div>
    <div class="stat"><div class="stat-label">Utilization</div><div class="stat-value">${res.utilization}%</div></div>
  `;

  // Preview
  const fileUrl=`/api/file?p=${enc(res.file)}`;
  $('previewCard').innerHTML=`
    <img src="${fileUrl}" alt="Combined batch">
    <div class="result-actions">
      <a class="dl-btn primary" href="${fileUrl}" download="${esc(res.filename)}">↓ Download Combined PNG</a>
    </div>
  `;

  // Packed image list
  if(res.images&&res.images.length){
    let html=`<div class="packed-list-title">Packed Assets (${res.images.length})</div>`;
    res.images.forEach(img=>{
      const thumbUrl=img.thumb?`/api/file?p=${enc(img.thumb)}`:'';
      const badge=img.had_alpha?'<span class="badge badge-blue">Had Alpha</span>':'<span class="badge badge-green">BG Removed</span>';
      html+=`<div class="packed-item">
        ${thumbUrl?`<img src="${thumbUrl}" alt="">`:''}
        <div class="pi-name">${esc(img.name)}${badge}</div>
        <div class="pi-size">${img.packed_size}</div>
      </div>`;
    });
    $('packedList').innerHTML=html;
  }
}

function toggleConsole(){
  $('consoleHead').classList.toggle('open');
  $('consoleBody').classList.toggle('open');
}
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    # ══════════════════════════════════════════════════════════════
    # LAUNCH (same robust pattern as trellis2_gui_process.py)
    # ══════════════════════════════════════════════════════════════
    import socket as _socket
    import requests as _requests

    def _find_free_port(preferred=5050):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", preferred))
                return preferred
            except OSError:
                s.bind(("0.0.0.0", 0))
                return s.getsockname()[1]

    PORT = _find_free_port(5050)

    _server_error = [None]

    def run_server():
        try:
            app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
        except Exception as exc:
            _server_error[0] = exc
            sys.__stderr__.write(f"\n❌ Flask server crashed: {exc}\n")
            traceback.print_exc(file=sys.__stderr__)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # ── Wait for Flask to respond ──
    _local_ready = False
    _last_local_err = None
    for _attempt in range(80):
        if _server_error[0] is not None:
            break
        try:
            _r = _requests.get(f"http://127.0.0.1:{PORT}/api/keepalive", timeout=0.75)
            if _r.status_code == 200:
                _local_ready = True
                break
        except Exception as _e:
            _last_local_err = _e
        time.sleep(0.5)

    if not _local_ready:
        raise RuntimeError(
            f"Flask server never became reachable on localhost:{PORT}.\n"
            f"  Server-thread error : {_server_error[0]}\n"
            f"  Last health-check error: {_last_local_err}"
        )

    sys.__stdout__.write(f"✅ Flask server healthy on localhost:{PORT}\n")

    # ── Obtain & verify public access (Colab only) ──
    _launch_mode = None
    public_url = None

    if IN_COLAB:
        from IPython.display import display, HTML as _HTML

        # Tier 1: proxyPort URL
        _last_proxy_err = None
        for _proxy_attempt in range(20):
            try:
                _candidate = eval_js(
                    f"google.colab.kernel.proxyPort({PORT}, {{'cache': false}})"
                )
                if _candidate and not _candidate.startswith("http"):
                    _candidate = "https://" + _candidate
            except Exception as _pe:
                _last_proxy_err = _pe
                _candidate = None

            if _candidate:
                try:
                    _rr = _requests.get(
                        _candidate.rstrip("/") + "/api/keepalive", timeout=4
                    )
                    if _rr.status_code == 200:
                        public_url = _candidate
                        _launch_mode = "proxy_url"
                        break
                except Exception:
                    pass
            time.sleep(0.5)

        if _launch_mode == "proxy_url":
            display(_HTML(f"""
            <div style="margin:16px 0;padding:16px 24px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;">
                <div style="color:#8A8A8A;font-size:13px;margin-bottom:8px;">🔺 Batch Image Packer is live — click to open:</div>
                <a href="{public_url}" target="_blank" style="color:#E8A917;font-size:18px;font-weight:bold;text-decoration:underline;">{public_url}</a>
                <div style="color:#8A8A8A;font-size:12px;margin-top:10px;">
                    Health check:
                    <a href="{public_url.rstrip('/')}/api/keepalive" target="_blank"
                       style="color:#8A8A8A;text-decoration:underline;">/api/keepalive</a>
                </div>
            </div>
            """))
        else:
            sys.__stdout__.write(
                f"⚠ Proxy URL not reachable (last error: {_last_proxy_err}).\n"
                f"  Falling back to embedded iframe...\n"
            )

            _POPOUT_BTN_JS = """
            <script>
            (function() {
                var btn = document.getElementById('packer-popout-btn');
                if (!btn) return;
                btn.addEventListener('click', async function() {
                    btn.textContent = 'Opening...';
                    btn.style.opacity = '0.6';
                    try {
                        var url = await google.colab.kernel.proxyPort(%d, {cache: false});
                        if (url && !url.startsWith('http')) url = 'https://' + url;
                        window.open(url, '_blank');
                        btn.textContent = '↗ Open in new tab';
                        btn.style.opacity = '1';
                    } catch(e) {
                        btn.textContent = '⚠ Failed — try again';
                        btn.style.opacity = '1';
                    }
                });
            })();
            </script>
            """ % PORT

            _POPOUT_BTN_HTML = (
                '<button id="packer-popout-btn" style="'
                'margin-left:12px;padding:6px 14px;'
                'background:#E8A917;color:#141414;border:none;border-radius:6px;'
                'font-family:monospace;font-size:13px;font-weight:bold;cursor:pointer;'
                '">↗ Open in new tab</button>'
            )

            _iframe_ok = False
            try:
                from google.colab import output as _colab_output
                _colab_output.serve_kernel_port_as_iframe(PORT, height='820')
                _launch_mode = "iframe"
                _iframe_ok = True
                display(_HTML(f"""
                <div style="margin:8px 0 4px;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                    <span style="color:#E8A917;font-weight:bold;">🔺 Batch Image Packer</span>
                    <span style="color:#8A8A8A;font-size:13px;"> — embedded above ↑</span>
                    {_POPOUT_BTN_HTML}
                </div>
                {_POPOUT_BTN_JS}
                """))
            except Exception as _iframe_err:
                sys.__stdout__.write(f"  ⚠ iframe fallback failed: {_iframe_err}\n")

            if not _iframe_ok:
                try:
                    from google.colab import output as _colab_output
                    _colab_output.serve_kernel_port_as_window(PORT, anchor_text="🔺 Click to open Batch Image Packer")
                    _launch_mode = "window"
                    display(_HTML(f"""
                    <div style="margin:8px 0;padding:10px 16px;background:#141414;border:2px solid #E8A917;border-radius:12px;font-family:monospace;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
                        <span style="color:#8A8A8A;font-size:13px;">Click the link above to open the UI, or:</span>
                        {_POPOUT_BTN_HTML}
                    </div>
                    {_POPOUT_BTN_JS}
                    """))
                except Exception as _window_err:
                    sys.__stdout__.write(f"  ⚠ window fallback also failed: {_window_err}\n")
                    try:
                        from IPython.display import Javascript as _JS
                        display(_JS("""
                        (async () => {
                            const url = await google.colab.kernel.proxyPort(%d, {cache: false});
                            const iframe = document.createElement('iframe');
                            iframe.src = url;
                            iframe.width = '100%%';
                            iframe.height = '820';
                            iframe.style.border = '2px solid #E8A917';
                            iframe.style.borderRadius = '12px';
                            document.querySelector('#output-area').appendChild(iframe);
                        })();
                        """ % PORT))
                        _launch_mode = "js_iframe"
                    except Exception as _js_err:
                        raise RuntimeError(
                            f"All Colab display methods failed for port {PORT}.\n"
                            f"  proxyPort error  : {_last_proxy_err}\n"
                            f"  iframe error     : {_iframe_err}\n"
                            f"  window error     : {_window_err}\n"
                            f"  JS iframe error  : {_js_err}\n"
                            f"  The Flask server IS running on localhost:{PORT}."
                        )

        sys.__stdout__.write(f"🚀 Launch mode: {_launch_mode}\n")

    else:
        _launch_mode = "local"
        print(f"\n🔺 Batch Image Packer running at http://localhost:{PORT}\n")

    print("Server running. Interrupt cell to stop.\n")

    try:
        _start_ts = time.time()
        while True:
            time.sleep(30)
            _uptime = int(time.time() - _start_ts)
            _h, _rem = divmod(_uptime, 3600)
            _m, _s = divmod(_rem, 60)
            sys.__stdout__.write(
                f"\r🔺 Uptime: {_h:02d}:{_m:02d}:{_s:02d} | Port: {PORT} | Mode: {_launch_mode} | Jobs: {len(jobs)}   "
            )
            sys.__stdout__.flush()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")

