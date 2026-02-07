from __future__ import annotations

import os
from pathlib import Path


def patch_transformers_missing_all_tied_weights_keys() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    # Only patch once
    if getattr(PreTrainedModel, "_trellis2_hf_patch_applied", False):
        return

    # --- Patch 1 ---
    orig_mark = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
    if orig_mark is not None and not hasattr(PreTrainedModel, "_trellis2_orig_mark_tied_weights_as_initialized"):
        PreTrainedModel._trellis2_orig_mark_tied_weights_as_initialized = orig_mark

        def _trellis2_safe_mark_tied_weights_as_initialized(self):
            if not hasattr(self, "all_tied_weights_keys"):
                return
            return PreTrainedModel._trellis2_orig_mark_tied_weights_as_initialized(self)

        PreTrainedModel.mark_tied_weights_as_initialized = _trellis2_safe_mark_tied_weights_as_initialized

    # --- Patch 2 ---
    orig_move = getattr(PreTrainedModel, "_move_missing_keys_from_meta_to_device", None)
    if orig_move is not None and not hasattr(PreTrainedModel, "_trellis2_orig_move_missing_keys_from_meta_to_device"):
        PreTrainedModel._trellis2_orig_move_missing_keys_from_meta_to_device = orig_move

        def _trellis2_safe_move_missing_keys_from_meta_to_device(
            self, missing_keys, device_map, device_mesh, hf_quantizer
        ):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return PreTrainedModel._trellis2_orig_move_missing_keys_from_meta_to_device(
                self, missing_keys, device_map, device_mesh, hf_quantizer
            )

        PreTrainedModel._move_missing_keys_from_meta_to_device = _trellis2_safe_move_missing_keys_from_meta_to_device

    PreTrainedModel._trellis2_hf_patch_applied = True

def ensure_patched_weights_snapshot() -> None:
    """
    Auto-download a TRELLIS weights snapshot to a local directory and patch text files inside it
    by replacing a DINO repo id string OLD -> NEW.

    Extra rule: ignore any file named `_colab_compat.py` anywhere in the tree.

    Env vars:
      TRELLIS2_PATCH_WEIGHTS        default "1"  (set "0" to disable)
      TRELLIS2_FORCE_PATCH_WEIGHTS  default "0"  (set "1" to re-run patch even if marker exists)
      TRELLIS2_WEIGHTS_REPO         default "microsoft/TRELLIS.2-4B"
      TRELLIS2_WEIGHTS_DIR          default "~/.cache/trellis2/weights"
      TRELLIS2_DINO_OLD             default "facebook/dinov3-vitl16-pretrain-lvd1689m"
      TRELLIS2_DINO_NEW             default "kiennt120/dinov3-vitl16-pretrain-lvd1689m"

    Side effect:
      Sets TRELLIS2_WEIGHTS_LOCAL_DIR to the patched local directory (if not already set).
    """
    if os.getenv("TRELLIS2_PATCH_WEIGHTS", "1") != "1":
        return

    # Only run once per process
    if getattr(ensure_patched_weights_snapshot, "_ran", False):
        return

    # Lazy import so your package can import even if HF hub isn't installed yet.
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        ensure_patched_weights_snapshot._ran = True
        return

    repo_id = os.getenv("TRELLIS2_WEIGHTS_REPO", "microsoft/TRELLIS.2-4B")
    local_dir = Path(
        os.getenv(
            "TRELLIS2_WEIGHTS_DIR",
            str(Path.home() / ".cache" / "trellis2" / "weights"),
        )
    ).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    old = os.getenv("TRELLIS2_DINO_OLD", "facebook/dinov3-vitl16-pretrain-lvd1689m")
    new = os.getenv("TRELLIS2_DINO_NEW", "kiennt120/dinov3-vitl16-pretrain-lvd1689m")

    marker = local_dir / ".trellis2_patched_marker"
    force = os.getenv("TRELLIS2_FORCE_PATCH_WEIGHTS", "0") == "1"

    # Download/refresh snapshot locally
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision="main",
    )

    # Patch text files once (idempotent)
    if (not marker.exists()) or force:
        bin_exts = {
            ".safetensors", ".bin", ".pt", ".pth",
            ".png", ".jpg", ".jpeg", ".webp", ".gif",
            ".mp4", ".mov", ".avi",
            ".glb", ".gltf", ".fbx", ".obj",
            ".onnx", ".npz",
        }

        modified_files = 0
        total_replacements = 0

        for p in local_dir.rglob("*"):
            if not p.is_file():
                continue

            # Ignore this filename anywhere in the tree
            if p.name == "_colab_compat.py":
                continue

            # Skip obvious binaries
            if p.suffix.lower() in bin_exts:
                continue

            try:
                b = p.read_bytes()
            except Exception:
                continue

            # Decode as utf-8 else latin-1, preserve chosen encoding on write
            try:
                t = b.decode("utf-8")
                enc = "utf-8"
            except UnicodeDecodeError:
                try:
                    t = b.decode("latin-1")
                    enc = "latin-1"
                except Exception:
                    continue

            if old not in t:
                continue

            c = t.count(old)
            try:
                p.write_bytes(t.replace(old, new).encode(enc))
            except Exception:
                continue

            modified_files += 1
            total_replacements += c

        marker.write_text(
            f"patched {old} -> {new}\n"
            f"modified_files={modified_files}\n"
            f"total_replacements={total_replacements}\n"
        )

    # Expose the patched directory for your loader
    os.environ.setdefault("TRELLIS2_WEIGHTS_LOCAL_DIR", str(local_dir))

    ensure_patched_weights_snapshot._ran = True
