from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils

# Patches Needed for colab compatibility
from ._colab_compat import (
    patch_transformers_missing_all_tied_weights_keys
)

patch_transformers_missing_all_tied_weights_keys()
# --- PATCH: provide a default all_tied_weights_keys for remote models that don't define it ---
import torch

if not hasattr(torch.nn.Module, "_patched_all_tied_weights_keys"):
    torch.nn.Module._patched_all_tied_weights_keys = True

    @property
    def _all_tied_weights_keys_default(self):
        # Transformers expects a dict-like with .keys(); empty means "no tied weights"
        return {}

    # Only define it if the attribute truly doesn't exist on the instance/class
    # By attaching to Module, any custom model (including remote ones) inherits it.
    setattr(torch.nn.Module, "all_tied_weights_keys", _all_tied_weights_keys_default)

# --- END PATCH ---

from huggingface_hub import snapshot_download
from pathlib import Path

SRC_REPO = "microsoft/TRELLIS.2-4B"
LOCAL_DIR = Path("/content/trellis2_weights_local")

OLD = "facebook/dinov3-vitl16-pretrain-lvd1689m"
NEW = "kiennt120/dinov3-vitl16-pretrain-lvd1689m"

print("Downloading weights repo locally...")
snapshot_download(
    repo_id=SRC_REPO,
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
    revision="main",
)

def is_text_file(p: Path) -> bool:
    # Simple heuristic: skip common binary extensions
    bin_exts = {".safetensors", ".bin", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".glb", ".onnx"}
    if p.suffix.lower() in bin_exts:
        return False
    return True

changed = []
for p in LOCAL_DIR.rglob("*"):
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

    if OLD not in t:
        continue

    c = t.count(OLD)
    p.write_bytes(t.replace(OLD, NEW).encode(enc))
    changed.append((str(p), c))

print(f"\nReplaced '{OLD}' -> '{NEW}' in {len(changed)} file(s).")
for fp, c in changed[:50]:
    print(f" - {fp} ({c}x)")
if len(changed) > 50:
    print(f" ... and {len(changed)-50} more")
