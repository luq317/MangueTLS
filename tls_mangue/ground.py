"""Etapa 2: solo robusto a lacunas (agua, canais de mare) e normalizacao de altura.

Metodo simples e transparente: percentil baixo de z por celula, remocao de celulas espurias
(reflexo na agua, pneumatoforos) por comparacao com a mediana local, preenchimento das lacunas por
interpolacao e suavizacao so nas celulas preenchidas.
Para producao, compare com o CSF (CloudCompare ou Sylva) e confira o resultado visualmente.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter, uniform_filter
from scipy.interpolate import griddata, RegularGridInterpolator


def build_dtm(xyz, p):
    """Retorna (funcao dtm(x, y) -> z do solo, relatorio)."""
    cell = p.ground_cell
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ix = ((xyz[:, 0] - x0) / cell).astype(int)
    iy = ((xyz[:, 1] - y0) / cell).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    df = pd.DataFrame({"ix": ix, "iy": iy, "z": xyz[:, 2]})
    g = df.groupby(["ix", "iy"])["z"]
    stat = g.agg(q=lambda s: np.percentile(s, p.ground_percentile), n="size").reset_index()
    stat = stat[stat["n"] >= p.ground_min_points]
    Z = np.full((nx, ny), np.nan)
    Z[stat["ix"].to_numpy(), stat["iy"].to_numpy()] = stat["q"].to_numpy()
    n_measured = int(np.isfinite(Z).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = generic_filter(Z, np.nanmedian, size=3, mode="nearest")
    spikes = np.isfinite(Z) & np.isfinite(med) & (np.abs(Z - med) > p.ground_spike_tol)
    Z2 = Z.copy()
    Z2[spikes] = np.nan
    # filtro regional (opcional): celulas so com dossel (sem retorno do chao, comum nas bordas do recorte) ficam
    # muito acima da mediana de uma janela grande e viram lacuna. Desligado quando ground_regional_tol <= 0.
    n_regional = 0
    if getattr(p, "ground_regional_tol", 0.0) > 0:
        R = max(3, int(p.ground_regional_cells) | 1)
        for _ in range(8):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medR = generic_filter(Z2, np.nanmedian, size=R, mode="nearest")
            bad = np.isfinite(Z2) & np.isfinite(medR) & (np.abs(Z2 - medR) > p.ground_regional_tol)
            if not bad.any():
                break
            n_regional += int(bad.sum()); Z2[bad] = np.nan
    valid = np.isfinite(Z2)
    if valid.sum() < 3:
        raise RuntimeError("Poucas celulas de solo validas. Revise ground_percentile, ground_min_points e a nuvem.")

    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    pts = np.column_stack([gx[valid], gy[valid]])
    vals = Z2[valid]
    filled = griddata(pts, vals, (gx, gy), method="linear")
    near = griddata(pts, vals, (gx, gy), method="nearest")
    filled = np.where(np.isfinite(filled), filled, near)
    smooth = uniform_filter(filled, size=max(1, p.ground_smooth_cells), mode="nearest")
    Zf = np.where(valid, Z2, smooth)
    if getattr(p, "ground_regional_tol", 0.0) > 0:
        lo, hi = np.percentile(Z2[valid], [2, 98])
        Zf = np.clip(Zf, lo - p.ground_regional_tol, hi + p.ground_regional_tol)  # celulas preenchidas nao saem da faixa do solo medido

    cx = x0 + (np.arange(nx) + 0.5) * cell
    cy = y0 + (np.arange(ny) + 0.5) * cell
    if nx == 1 or ny == 1:
        raise RuntimeError("Parcela menor que uma celula do solo. Reduza ground_cell.")
    interp = RegularGridInterpolator((cx, cy), Zf, bounds_error=False, fill_value=None)

    def dtm(x, y):
        return interp(np.column_stack([x, y]))

    report = {
        "dtm_cells": int(nx * ny),
        "dtm_cells_measured": n_measured,
        "dtm_cells_spike_removed": int(spikes.sum()),
        "dtm_cells_regional_removed": n_regional,
        "dtm_gap_fraction": float(1 - valid.sum() / (nx * ny)),
        "dtm_z_min": float(Zf.min()), "dtm_z_max": float(Zf.max()),
    }
    return dtm, report, (cx, cy, Zf)


def normalize_height(xyz, dtm, p):
    """Altura acima do solo; descarta pontos abaixo de below_ground_cut. Retorna (z_norm, mascara de pontos mantidos)."""
    zn = xyz[:, 2] - dtm(xyz[:, 0], xyz[:, 1])
    keep = zn >= p.below_ground_cut
    return zn, keep
