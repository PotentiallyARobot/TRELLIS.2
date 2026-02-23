from typing import *
import torch
import torch.nn as nn
from .. import models


def _low_memory_download(repo_id: str, local_dir: str) -> str:
    """
    Download a HuggingFace repo one file at a time to avoid RAM spikes.

    The default snapshot_download uses multiple parallel workers that each
    buffer aggressively in RAM. On machines with limited system memory
    (e.g. Colab T4 free tier with ~12.7 GB) this causes OOM kills during
    the 16+ GB TRELLIS.2 download.

    This function downloads files sequentially with gc.collect() between
    each file, keeping peak RAM to roughly one file's buffer size (~500 MB).
    """
    import os
    import gc
    import sys
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id)
    total = len(repo_files)

    for i, fname in enumerate(repo_files, 1):
        dest = local / fname
        if dest.exists() and dest.stat().st_size > 0:
            sys.stderr.write(f"\r  [{i}/{total}] {fname} — cached      ")
            sys.stderr.flush()
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(f"\r  [{i}/{total}] {fname}...")
        sys.stderr.flush()

        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(local),
                local_dir_use_symlinks=False,
                force_download=False,
            )
        except Exception:
            # Retry once with force
            gc.collect()
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=str(local),
                    local_dir_use_symlinks=False,
                    force_download=True,
                )
            except Exception as e2:
                sys.stderr.write(f" ❌ {e2}\n")
                raise

        gc.collect()

    sys.stderr.write(f"\n  ✅ {total} files downloaded to {local}\n")
    sys.stderr.flush()
    return str(local)


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        # ── If path is a HuggingFace repo, download everything locally first ──
        # This avoids per-model HF downloads that each spawn parallel workers
        # and spike RAM. One sequential download, then all-local loading.
        if not is_local:
            # Check for an environment-provided local cache first
            # (set by the GUI wrapper's resolve_weights)
            env_local = os.environ.get("TRELLIS2_LOCAL_WEIGHTS")
            if env_local and os.path.exists(f"{env_local}/{config_file}"):
                path = env_local
                is_local = True
            else:
                # Download the entire repo one file at a time
                import tempfile
                default_local = os.path.join(
                    tempfile.gettempdir(), "trellis2_weights"
                )
                local_dir = os.environ.get("TRELLIS2_DOWNLOAD_DIR", default_local)
                print(f"⬇ Downloading {path} → {local_dir} (low-memory, one file at a time)...")
                path = _low_memory_download(path, local_dir)
                is_local = True

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
