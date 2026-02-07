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

