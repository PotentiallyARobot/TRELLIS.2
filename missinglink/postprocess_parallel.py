# ============================================================
# postprocess_parallel.py  v2 — Pipelined Edition
# Drop-in replacement for o_voxel.postprocess.to_glb that splits
# the pipeline into 3 phases so xatlas (CPU) can run in parallel
# across multiple meshes while GPU does reconstruction.
#
# USAGE — paste this file into /content/ alongside the notebook.
#
# Phase 1: prepare_mesh()  — GPU: cleanup, remesh, simplify, BVH build
# Phase 2: uv_unwrap()     — CPU: xatlas charting (THE BOTTLENECK) ← PARALLEL
# Phase 3: bake_and_export() — GPU: rasterize UVs, sample attrs, write GLB
# ============================================================

from typing import *
from dataclasses import dataclass, field
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
import time
import os


@dataclass
class PreparedMesh:
    """Output of Phase 1 — everything needed for UV unwrap + bake."""
    cumesh_obj: Any = None

    orig_vertices: torch.Tensor = None
    orig_faces: torch.Tensor = None
    bvh: Any = None

    attr_volume: torch.Tensor = None
    coords: torch.Tensor = None
    attr_layout: Dict = field(default_factory=dict)
    aabb: torch.Tensor = None
    voxel_size: torch.Tensor = None
    grid_size: torch.Tensor = None

    texture_size: int = 2048
    remesh: bool = False
    name: str = ""


@dataclass
class UnwrappedMesh:
    """Output of Phase 2 — UV-unwrapped mesh ready for GPU bake."""
    out_vertices: torch.Tensor = None
    out_faces: torch.Tensor = None
    out_uvs: torch.Tensor = None
    out_normals: torch.Tensor = None

    orig_vertices: torch.Tensor = None
    orig_faces: torch.Tensor = None
    bvh: Any = None

    attr_volume: torch.Tensor = None
    coords: torch.Tensor = None
    attr_layout: Dict = field(default_factory=dict)
    aabb: torch.Tensor = None
    voxel_size: torch.Tensor = None
    grid_size: torch.Tensor = None

    texture_size: int = 2048
    remesh: bool = False
    name: str = ""


# ═══════════════════════════════════════════════════════════════
# Phase 1: prepare_mesh — GPU bound (~3-10s)
# ═══════════════════════════════════════════════════════════════

def prepare_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb,
    voxel_size=None,
    grid_size=None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    verbose: bool = False,
    name: str = "",
) -> PreparedMesh:
    """
    Phase 1: GPU — clean, remesh, simplify, build BVH.
    Returns a PreparedMesh with cumesh object ready for uv_unwrap.
    """
    import cumesh

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    if verbose:
        print(f"  [prepare] Original: {vertices.shape[0]:,} verts, {faces.shape[0]:,} faces")

    vertices = vertices.cuda()
    faces = faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()

    bvh = cumesh.cuBVH(vertices, faces)

    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=verbose,
            bvh=bvh,
        ))
        if verbose:
            print(f"  [prepare] After remesh: {mesh.num_vertices:,} verts, {mesh.num_faces:,} faces")
        mesh.simplify(decimation_target, verbose=verbose)

    if verbose:
        print(f"  [prepare] Final: {mesh.num_vertices:,} verts, {mesh.num_faces:,} faces")

    torch.cuda.synchronize()

    return PreparedMesh(
        cumesh_obj=mesh,
        orig_vertices=vertices,
        orig_faces=faces,
        bvh=bvh,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        texture_size=texture_size,
        remesh=remesh,
        name=name,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 2: uv_unwrap — CPU bound (xatlas) ← RUN IN PARALLEL
# ═══════════════════════════════════════════════════════════════

def uv_unwrap(
    prepared: PreparedMesh,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
) -> UnwrappedMesh:
    """
    Phase 2: xatlas UV unwrap — CPU-bound.
    SAFE TO RUN FROM MULTIPLE THREADS — xatlas instances are independent.
    """
    mesh = prepared.cumesh_obj
    t0 = time.perf_counter()

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )

    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    dt = time.perf_counter() - t0
    if verbose:
        print(f"  [uv_unwrap] {prepared.name}: {dt:.1f}s "
              f"({out_vertices.shape[0]:,} verts, {out_faces.shape[0]:,} faces)")

    return UnwrappedMesh(
        out_vertices=out_vertices.cpu(),
        out_faces=out_faces.cpu(),
        out_uvs=out_uvs.cpu(),
        out_normals=out_normals.cpu(),
        orig_vertices=prepared.orig_vertices,
        orig_faces=prepared.orig_faces,
        bvh=prepared.bvh,
        attr_volume=prepared.attr_volume,
        coords=prepared.coords,
        attr_layout=prepared.attr_layout,
        aabb=prepared.aabb,
        voxel_size=prepared.voxel_size,
        grid_size=prepared.grid_size,
        texture_size=prepared.texture_size,
        remesh=prepared.remesh,
        name=prepared.name,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 3: bake_and_export — GPU bound (~5-10s)
# ═══════════════════════════════════════════════════════════════

def bake_and_export(
    unwrapped: UnwrappedMesh,
    output_path: str,
    verbose: bool = False,
) -> str:
    """
    Phase 3: GPU — rasterize UVs, sample volume attributes, inpaint, write GLB.
    Must be called ONE AT A TIME (serialized) because nvdiffrast + triton
    are not thread-safe.
    """
    from flex_gemm.ops.grid_sample import grid_sample_3d
    import nvdiffrast.torch as dr

    t0 = time.perf_counter()
    texture_size = unwrapped.texture_size

    out_vertices = unwrapped.out_vertices.cuda()
    out_faces = unwrapped.out_faces.cuda()
    out_uvs = unwrapped.out_uvs.cuda()
    out_normals = unwrapped.out_normals

    attr_volume = unwrapped.attr_volume
    coords = unwrapped.coords
    aabb = unwrapped.aabb
    voxel_size = unwrapped.voxel_size
    grid_size = unwrapped.grid_size

    if not attr_volume.is_cuda:
        attr_volume = attr_volume.cuda()
    if not coords.is_cuda:
        coords = coords.cuda()
    if not aabb.is_cuda:
        aabb = aabb.cuda()
    if not voxel_size.is_cuda:
        voxel_size = voxel_size.cuda()

    orig_vertices = unwrapped.orig_vertices
    orig_faces = unwrapped.orig_faces
    if not orig_vertices.is_cuda:
        orig_vertices = orig_vertices.cuda()
    if not orig_faces.is_cuda:
        orig_faces = orig_faces.cuda()
    bvh = unwrapped.bvh

    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([
        out_uvs * 2 - 1,
        torch.zeros_like(out_uvs[:, :1]),
        torch.ones_like(out_uvs[:, :1])
    ], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)

    mask = rast[0, ..., 3] > 0

    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]

    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )

    mask_np = mask.cpu().numpy()
    attr_layout = unwrapped.attr_layout

    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    mask_inv = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=not unwrapped.remesh,
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.numpy() if not out_normals.is_cuda else out_normals.cpu().numpy()

    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )

    textured_mesh.export(str(output_path), extension_webp=True)
    torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    if verbose:
        sz = os.path.getsize(output_path)
        print(f"  [bake] {unwrapped.name}: {dt:.1f}s → {sz / 1048576:.1f} MB")

    del attrs, rast, pos, valid_pos, out_vertices, out_faces, out_uvs
    torch.cuda.empty_cache()

    return str(output_path)
