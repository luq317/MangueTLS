"""Etapa 1: voxelizacao e remocao de ruido (SOR e ROR)."""
import numpy as np
from scipy.spatial import cKDTree


def voxel_indices(xyz, voxel):
    """Indices de um ponto por voxel (o primeiro encontrado). voxel <= 0 mantem todos."""
    if voxel <= 0:
        return np.arange(len(xyz))
    k = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    m = k.max(axis=0) + 1
    key = (k[:, 0] * m[1] + k[:, 1]) * m[2] + k[:, 2]
    _, idx = np.unique(key, return_index=True)
    return np.sort(idx)


def sor_mask(xyz, k=8, std_ratio=2.0):
    """Statistical Outlier Removal. Limiar global: rode por scan ou depois do voxel quando a densidade variar muito."""
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1, workers=-1)
    md = d[:, 1:].mean(axis=1)
    return md <= md.mean() + std_ratio * md.std()


def ror_mask(xyz, radius=0.10, min_neighbors=3):
    """Radius Outlier Removal."""
    tree = cKDTree(xyz)
    cnt = tree.query_ball_point(xyz, r=radius, return_length=True, workers=-1)
    return (cnt - 1) >= min_neighbors


def preprocess(xyz, p):
    """Voxel, SOR e ROR em sequencia. Retorna (xyz_filtrado, relatorio)."""
    n0 = len(xyz)
    xyz = xyz[voxel_indices(xyz, p.voxel_size)]
    n1 = len(xyz)
    xyz = xyz[sor_mask(xyz, p.sor_k, p.sor_std)]
    n2 = len(xyz)
    xyz = xyz[ror_mask(xyz, p.ror_radius, p.ror_min_neighbors)]
    return xyz, {"n_input": n0, "n_voxel": n1, "n_sor": n2, "n_ror": len(xyz)}
