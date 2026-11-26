"""Etapa 5: agregacao por parcela."""
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside(poly, x, y):
    return MplPath(poly).contains_points(np.column_stack([x, y]))


def aggregate(stems_df, area_m2, min_quality=0.0, min_dbh_cm=0.0, label="all"):
    """DAP medio (cm), altura media (m), densidade (troncos/ha) e area basal (m2/ha) de uma parcela.
    Cada fuste conta como um tronco (como no inventario do NEMA, que mede cada tronco).
    O status vivo/morto nao e detectado: se a tabela tiver a coluna 'status', so 'live' entra."""
    d = stems_df
    if "status" in d.columns:
        d = d[d["status"].astype(str).str.lower().isin(["live", "vivo", "alive"])]
    d = d[(d["quality"] >= min_quality) & (d["dbh_cm"] >= min_dbh_cm)]
    n = len(d)
    ha = area_m2 / 1e4
    return {
        "subset": label, "n_stems": n, "area_m2": area_m2,
        "dbh_mean_cm": float(d["dbh_cm"].mean()) if n else np.nan,
        "height_mean_m": float(d["height_m"].mean()) if n else np.nan,
        "density_ha": n / ha if ha > 0 else np.nan,
        "basal_area_m2ha": float(d["basal_area_m2"].sum() / ha) if n else np.nan,
    }


def hull_area(xy):
    return float(ConvexHull(xy).volume)
