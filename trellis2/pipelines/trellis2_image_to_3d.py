import os
import time
import pickle
from typing import *
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


# ═══════════════════════════════════════════════════════════════════════════
# Stage Cache Helper
# ═══════════════════════════════════════════════════════════════════════════

class StageCache:
    """
    Saves / loads intermediate pipeline tensors to disk.
    SparseTensors are stored as plain dicts {feats, coords} so they can
    be loaded and inspected without the full trellis2 stack.
    """

    STAGE_ORDER = [
        "image_cond_512",
        "image_cond_1024",
        "tex_cond_512",
        "tex_cond_1024",
        "sparse_structure",
        "shape_slat",
        "tex_slat",
        "decoded_mesh",
    ]

    STAGE_DESCRIPTIONS = {
        "image_cond_512": (
            "DINO conditioning @ 512px (weighted blend of input images).\n"
            "  dict {'cond': (1,N,D), 'neg_cond': (1,N,D)}.\n"
            "  Used by: sparse structure sampler, shape flow (512 pipeline)."
        ),
        "image_cond_1024": (
            "DINO conditioning @ 1024px (weighted blend of input images).\n"
            "  dict {'cond': (1,N,D), 'neg_cond': (1,N,D)}.\n"
            "  Used by: shape flow (cascade/1024), texture flow."
        ),
        "tex_cond_512": (
            "Texture DINO conditioning @ 512px (may use tex_image_weights).\n"
            "  Only saved when tex_image_weights differs from image_weights."
        ),
        "tex_cond_1024": (
            "Texture DINO conditioning @ 1024px (may use tex_image_weights).\n"
            "  Only saved when tex_image_weights differs from image_weights."
        ),
        "sparse_structure": (
            "Occupancy coords (N,4) int tensor [batch, x, y, z].\n"
            "  Defines WHICH voxels exist — the silhouette/volume.\n"
            "  Reuse to keep consistent structure across different images."
        ),
        "shape_slat": (
            "Shape structured latent. dict {'feats': (N,C), 'coords': (N,4), 'resolution': int}.\n"
            "  Full geometric detail. Reuse to keep geometry while re-texturing."
        ),
        "tex_slat": (
            "Texture structured latent. dict {'feats': (N,C), 'coords': (N,4)}.\n"
            "  PBR appearance. Swap to re-skin the same shape."
        ),
        "decoded_mesh": (
            "Final MeshWithVoxel list (pickled).\n"
            "  The output you feed to o_voxel.postprocess.to_glb()."
        ),
    }

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.saved_stages: List[str] = []
        self.timings: Dict[str, float] = {}

    # ── Serialisation helpers ──────────────────────────────────────────────

    @staticmethod
    def sparse_to_dict(st) -> dict:
        """SparseTensor → portable dict."""
        return {"feats": st.feats.cpu(), "coords": st.coords.cpu(), "_type": "SparseTensor"}

    @staticmethod
    def dict_to_sparse(d: dict, device: str = "cuda"):
        """Portable dict → SparseTensor."""
        return SparseTensor(feats=d["feats"].to(device), coords=d["coords"].to(device))

    # ── Save / Load ────────────────────────────────────────────────────────

    def save(self, name: str, data: Any) -> Optional[Path]:
        if not self.enabled:
            return None
        path = self.cache_dir / f"{name}.pt"

        # SparseTensor (duck-typed: has .feats and .coords but is not a dict)
        if hasattr(data, "feats") and hasattr(data, "coords") and not isinstance(data, dict):
            torch.save(self.sparse_to_dict(data), path)

        # (SparseTensor, resolution) tuple from cascade
        elif isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], "feats"):
            torch.save({"slat": self.sparse_to_dict(data[0]), "resolution": data[1]}, path)

        # dict of tensors (cond dicts)
        elif isinstance(data, dict):
            torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}, path)

        # raw tensor (coords)
        elif isinstance(data, torch.Tensor):
            torch.save(data.cpu(), path)

        # list (meshes) → pickle
        elif isinstance(data, list):
            path = path.with_suffix(".pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)

        else:
            torch.save(data, path)

        self.saved_stages.append(name)
        sz = path.stat().st_size / 1048576
        print(f"  💾 cached {name} → {path}  ({sz:.1f} MB)")
        return path

    @staticmethod
    def load(path: str, as_sparse: bool = False, device: str = "cuda") -> Any:
        """Load a cached stage file."""
        p = Path(path)
        if p.suffix == ".pkl":
            with open(p, "rb") as f:
                return pickle.load(f)

        data = torch.load(p, map_location=device, weights_only=False)

        # SparseTensor dict
        if as_sparse and isinstance(data, dict) and data.get("_type") == "SparseTensor":
            return StageCache.dict_to_sparse(data, device)

        # Cascade tuple: {slat: dict, resolution: int}
        if isinstance(data, dict) and "slat" in data and "resolution" in data:
            if as_sparse:
                return (StageCache.dict_to_sparse(data["slat"], device), data["resolution"])
            return data

        # Move tensors back to device
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
        elif isinstance(data, torch.Tensor):
            data = data.to(device)

        return data

    def write_summary(self, meta: dict = None):
        """Write summary.txt describing all cached stages."""
        meta = meta or {}
        lines = [
            "TRELLIS.2 Stage Cache Summary",
            "=" * 60,
            f"Timestamp     : {datetime.now().isoformat()}",
            f"Cache dir     : {self.cache_dir}",
            f"Pipeline type : {meta.get('pipeline_type', 'N/A')}",
            f"Seed          : {meta.get('seed', 'N/A')}",
            f"Image(s)      : {meta.get('images', 'N/A')}",
            f"image_weights : {meta.get('image_weights', 'N/A')}",
            f"tex_weights   : {meta.get('tex_image_weights', 'N/A')}",
            "",
            "Cached Stages",
            "-" * 60,
        ]
        for stage in self.STAGE_ORDER:
            cached = "YES" if stage in self.saved_stages else "-- "
            pt = self.cache_dir / f"{stage}.pt"
            pkl = self.cache_dir / f"{stage}.pkl"
            ep = pt if pt.exists() else (pkl if pkl.exists() else None)
            sz = f"  ({ep.stat().st_size / 1048576:.1f} MB)" if ep else ""
            lines.append(f"  [{cached}] {stage}{sz}")
            for dl in self.STAGE_DESCRIPTIONS.get(stage, "").split('\n'):
                lines.append(f"        {dl}")
            lines.append("")

        lines += [
            "Re-use Examples",
            "-" * 60,
            "  # Keep structure, regenerate shape+texture with new image:",
            f"  load_stages={{'sparse_structure': '{self.cache_dir}/sparse_structure.pt'}}",
            "",
            "  # Keep shape geometry, regenerate texture only:",
            f"  load_stages={{'shape_slat': '{self.cache_dir}/shape_slat.pt'}}",
            "",
            "  # Keep everything except texture (re-skin with new image):",
            f"  load_stages={{",
            f"      'sparse_structure': '{self.cache_dir}/sparse_structure.pt',",
            f"      'shape_slat': '{self.cache_dir}/shape_slat.pt',",
            f"  }}",
        ]

        (self.cache_dir / "summary.txt").write_text('\n'.join(lines))
        print(f"  📝 summary → {self.cache_dir / 'summary.txt'}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Supports:
      - Multi-image weighted conditioning (image_weights, tex_image_weights)
      - Stage caching via cache_stages / load_stages params in run()
      - Background removal disable via TRELLIS2_DISABLE_REMBG env var
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])

        disable_rembg = os.getenv("TRELLIS2_DISABLE_REMBG", "0").lower() in ("1", "true", "yes")
        if disable_rembg:
            pipeline.rembg_model = None
        else:
            pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])

        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """Preprocess the input image."""
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        elif self.rembg_model is None:
            output = input.convert('RGBA')
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    # ── Conditioning ───────────────────────────────────────────────────────

    def get_cond_weighted(
            self,
            images: List[Image.Image],
            resolution: int,
            weights: Optional[List[float]] = None,
            include_neg_cond: bool = True,
    ) -> dict:
        """
        Extract DINO features for multiple images and reduce (B,N,D) → (1,N,D)
        via weighted average over B.
        """
        assert len(images) >= 1, "Need at least one image"
        if weights is None:
            weights = [1.0 / len(images)] * len(images)
        assert len(weights) == len(images), "weights must match images length"

        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        w = w / w.sum()

        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        feats = self.image_cond_model(images)
        if self.low_vram:
            self.image_cond_model.cpu()

        w = w.view(-1, 1, 1).to(feats.dtype).to(feats.device)
        cond = (feats * w).sum(dim=0, keepdim=True)

        if not include_neg_cond:
            return {"cond": cond}
        neg_cond = torch.zeros_like(cond)
        return {"cond": cond, "neg_cond": neg_cond}

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """Get conditioning information (single-image, kept for backwards compat)."""
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {'cond': cond, 'neg_cond': neg_cond}

    # ── Sparse Structure ───────────────────────────────────────────────────

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise,
            **cond, **sampler_params,
            verbose=True, tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    # ── Shape SLat ─────────────────────────────────────────────────────────

    def sample_shape_slat(
        self, cond: dict, flow_model,
        coords: torch.Tensor, sampler_params: dict = {},
    ) -> SparseTensor:
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model, noise,
            **cond, **sampler_params,
            verbose=True, tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    def sample_shape_slat_cascade(
        self, lr_cond: dict, cond: dict,
        flow_model_lr, flow_model,
        lr_resolution: int, resolution: int,
        coords: torch.Tensor, sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        # LR pass
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr, noise,
            **lr_cond, **sampler_params,
            verbose=True, tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128

        # HR pass
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model, noise,
            **cond, **sampler_params,
            verbose=True, tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat, hr_resolution

    # ── Decode ─────────────────────────────────────────────────────────────

    def decode_shape_slat(self, slat: SparseTensor, resolution: int) -> Tuple[List[Mesh], List[SparseTensor]]:
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret

    def sample_tex_slat(
        self, cond: dict, flow_model,
        shape_slat: SparseTensor, sampler_params: dict = {},
    ) -> SparseTensor:
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model, noise,
            concat_cond=shape_slat,
            **cond, **sampler_params,
            verbose=True, tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    def decode_tex_slat(self, slat: SparseTensor, subs: List[SparseTensor]) -> SparseTensor:
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh

    # ═══════════════════════════════════════════════════════════════════════
    # run() — with stage caching support
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def run(
            self,
            image: Union[Image.Image, List[Image.Image]],
            num_samples: int = 1,
            seed: int = 42,
            sparse_structure_sampler_params: dict = {},
            shape_slat_sampler_params: dict = {},
            tex_slat_sampler_params: dict = {},
            preprocess_image: bool = True,
            return_latent: bool = False,
            pipeline_type: Optional[str] = None,
            max_num_tokens: int = 49152,
            image_weights: Optional[List[float]] = None,
            tex_image_weights: Optional[List[float]] = None,
            # ── Stage caching (new) ──
            cache_stages: Optional[Union[str, "StageCache"]] = None,
            load_stages: Optional[Dict[str, str]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image: A single PIL image or a list of PIL images.
            num_samples: Number of 3D samples to generate.
            seed: RNG seed.
            sparse_structure_sampler_params: Overrides for sparse structure sampler.
            shape_slat_sampler_params: Overrides for shape sampler.
            tex_slat_sampler_params: Overrides for texture sampler.
            preprocess_image: Whether to preprocess each image.
            return_latent: Whether to return (meshes, (shape_slat, tex_slat, res)).
            pipeline_type: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens: Max tokens for cascade.
            image_weights: Weights for multi-image conditioning (shape/sparse).
            tex_image_weights: Separate weights for texture conditioning.

            cache_stages: Enable stage caching.
                - str: path to cache directory (creates StageCache automatically)
                - StageCache instance: use directly
                - None: disabled (default — behaves identically to before)
            load_stages: Dict mapping stage names to .pt file paths to load
                from disk instead of computing.  Example:
                    load_stages={'sparse_structure': 'cache_run1/sparse_structure.pt',
                                 'shape_slat':      'cache_run1/shape_slat.pt'}
        """
        t0 = time.perf_counter()

        # ── Setup cache ───────────────────────────────────────────────────
        cache: Optional[StageCache] = None
        if isinstance(cache_stages, str):
            cache = StageCache(cache_stages)
        elif isinstance(cache_stages, StageCache):
            cache = cache_stages
        load = load_stages or {}

        # ── Validate pipeline type ────────────────────────────────────────
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'tex_slat_flow_model_512' in self.models
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # ── Normalize inputs ──────────────────────────────────────────────
        images: List[Image.Image] = image if isinstance(image, list) else [image]
        if len(images) == 0:
            raise ValueError("No images provided.")
        if preprocess_image:
            images = [self.preprocess_image(im) for im in images]

        torch.manual_seed(seed)
        has_separate_tex_weights = (tex_image_weights is not None and tex_image_weights != image_weights)

        # ══════════════════════════════════════════════════════════════════
        # Stage 1: Image conditioning 512 (shape/structure)
        # ══════════════════════════════════════════════════════════════════
        if "image_cond_512" in load:
            print("[stage] Loading image_cond_512 from cache")
            cond_512 = StageCache.load(load["image_cond_512"])
        else:
            cond_512 = self.get_cond_weighted(images, 512, weights=image_weights)
        if cache:
            cache.save("image_cond_512", cond_512)

        # ══════════════════════════════════════════════════════════════════
        # Stage 2: Image conditioning 1024 (shape/structure)
        # ══════════════════════════════════════════════════════════════════
        cond_1024 = None
        if pipeline_type != '512':
            if "image_cond_1024" in load:
                print("[stage] Loading image_cond_1024 from cache")
                cond_1024 = StageCache.load(load["image_cond_1024"])
            else:
                cond_1024 = self.get_cond_weighted(images, 1024, weights=image_weights)
            if cache:
                cache.save("image_cond_1024", cond_1024)

        # ══════════════════════════════════════════════════════════════════
        # Stage 2b: Texture conditioning (separate weights if provided)
        # ══════════════════════════════════════════════════════════════════
        tex_w = tex_image_weights if tex_image_weights is not None else image_weights

        if "tex_cond_512" in load:
            tex_cond_512 = StageCache.load(load["tex_cond_512"])
        elif pipeline_type == '512':
            tex_cond_512 = self.get_cond_weighted(images, 512, weights=tex_w) if has_separate_tex_weights else cond_512
        else:
            tex_cond_512 = None

        if "tex_cond_1024" in load:
            tex_cond_1024 = StageCache.load(load["tex_cond_1024"])
        elif pipeline_type != '512':
            tex_cond_1024 = self.get_cond_weighted(images, 1024, weights=tex_w) if has_separate_tex_weights else cond_1024
        else:
            tex_cond_1024 = None

        if cache and has_separate_tex_weights:
            if tex_cond_512 is not None:
                cache.save("tex_cond_512", tex_cond_512)
            if tex_cond_1024 is not None:
                cache.save("tex_cond_1024", tex_cond_1024)

        # ══════════════════════════════════════════════════════════════════
        # Stage 3: Sparse structure
        # ══════════════════════════════════════════════════════════════════
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        if "sparse_structure" in load:
            print("[stage] Loading sparse_structure from cache")
            coords = StageCache.load(load["sparse_structure"])
            if isinstance(coords, torch.Tensor):
                coords = coords.to(self.device)
        else:
            coords = self.sample_sparse_structure(
                cond_512, ss_res, num_samples, sparse_structure_sampler_params
            )
        if cache:
            cache.save("sparse_structure", coords)

        # ══════════════════════════════════════════════════════════════════
        # Stage 4: Shape structured latent
        # ══════════════════════════════════════════════════════════════════
        res = None
        if "shape_slat" in load:
            print("[stage] Loading shape_slat from cache")
            loaded = StageCache.load(load["shape_slat"], as_sparse=True)
            if isinstance(loaded, tuple):
                shape_slat, res = loaded
            else:
                shape_slat = loaded
        else:
            if pipeline_type == '512':
                shape_slat = self.sample_shape_slat(
                    cond_512, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params)
                res = 512
            elif pipeline_type == '1024':
                shape_slat = self.sample_shape_slat(
                    cond_1024, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params)
                res = 1024
            elif pipeline_type == '1024_cascade':
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1024, coords, shape_slat_sampler_params, max_num_tokens)
            elif pipeline_type == '1536_cascade':
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1536, coords, shape_slat_sampler_params, max_num_tokens)

        if res is None:
            res = {'512': 512, '1024': 1024, '1024_cascade': 1024, '1536_cascade': 1536}[pipeline_type]
        if cache:
            cache.save("shape_slat", (shape_slat, res))

        # ══════════════════════════════════════════════════════════════════
        # Stage 5: Texture structured latent
        # ══════════════════════════════════════════════════════════════════
        if "tex_slat" in load:
            print("[stage] Loading tex_slat from cache")
            tex_slat = StageCache.load(load["tex_slat"], as_sparse=True)
        else:
            if pipeline_type == '512':
                tex_slat = self.sample_tex_slat(
                    tex_cond_512, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params)
            else:
                tex_slat = self.sample_tex_slat(
                    tex_cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params)
        if cache:
            cache.save("tex_slat", tex_slat)

        # ══════════════════════════════════════════════════════════════════
        # Stage 6: Decode → mesh
        # ══════════════════════════════════════════════════════════════════
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if cache:
            cache.save("decoded_mesh", out_mesh)
            cache.write_summary({
                'pipeline_type': pipeline_type,
                'seed': seed,
                'images': f"{len(images)} image(s)",
                'image_weights': image_weights,
                'tex_image_weights': tex_image_weights,
            })
            dt = time.perf_counter() - t0
            print(f"  ⏱️  Total pipeline time: {dt:.1f}s")

        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh
