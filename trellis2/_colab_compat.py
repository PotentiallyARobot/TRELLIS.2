from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

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

def download_and_patch_trellis_weights(
    local_dir: str | Path = "/content/trellis2_weights_local",
    src_repo: str = "microsoft/TRELLIS.2-4B",
    revision: str = "main",
    old: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    new: str = "kiennt120/dinov3-vitl16-pretrain-lvd1689m",
    ignore_filenames: tuple[str, ...] = ("_colab_compat.py",),
    force_redownload: bool = False,
    force_repatch: bool = False,
) -> tuple[Path, List[Tuple[str, int]]]:
    """
    Downloads the TRELLIS weights snapshot into `local_dir` and patches text files
    by replacing `old` -> `new`.

    Returns:
      (local_dir_path, changed_list)
        - local_dir_path: Path to the local snapshot directory
        - changed_list: list of (filepath, count_replacements_in_file)

    Behavior:
      - Uses Hugging Face snapshot_download.
      - Skips binary-ish extensions.
      - Skips any file whose name is in ignore_filenames (anywhere in tree).
      - Writes a marker file so patching is idempotent unless force_repatch=True.
      - Sets env var TRELLIS2_WEIGHTS_LOCAL_DIR to the snapshot path.
    """
    # Lazy import so this function can live in your repo without hard dependency at import time
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(os.getcwd())
    marker = "trellis2_weights_local"
    if marker.exists() and not force_repatch:
        # Still ensure env var is set, and optionally redownload if requested
        if force_redownload:
            snapshot_download(
                repo_id=src_repo,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                revision=revision,
                force_download=True,
            )
        os.environ.setdefault("TRELLIS2_WEIGHTS_LOCAL_DIR", str(local_dir))
        return local_dir, []

    print(f"Downloading weights repo locally to: {local_dir}")
    snapshot_download(
        repo_id=src_repo,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        force_download=force_redownload,
    )

    bin_exts = {
        ".safetensors", ".bin", ".pt", ".pth",
        ".png", ".jpg", ".jpeg", ".webp", ".gif",
        ".mp4", ".mov", ".avi",
        ".glb", ".gltf", ".fbx", ".obj",
        ".onnx", ".npz",
    }

    def is_text_file(p: Path) -> bool:
        if p.name in ignore_filenames:
            return False
        if p.suffix.lower() in bin_exts:
            return False
        return True

    changed: List[Tuple[str, int]] = []
    for p in local_dir.rglob("*"):
        if not p.is_file() or not is_text_file(p):
            continue

        try:
            b = p.read_bytes()
        except Exception:
            continue

        # Try utf-8 then latin-1
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

        changed.append((str(p), c))

    marker.write_text(
        f"patched: {old} -> {new}\n"
        f"files_changed: {len(changed)}\n"
        f"total_replacements: {sum(c for _, c in changed)}\n"
    )

    os.environ.setdefault("TRELLIS2_WEIGHTS_LOCAL_DIR", str(local_dir))

    print(f"\nReplaced '{old}' -> '{new}' in {len(changed)} file(s).")
    return local_dir, changed

