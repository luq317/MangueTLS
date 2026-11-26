"""Leitura e escrita de nuvens de pontos."""
import pathlib
import numpy as np
import pandas as pd


def thin_voxel(xyz, size):
    """Mantem 1 ponto por voxel (economiza memoria logo na leitura). Usa float32 e chaves int64."""
    mn = xyz.min(axis=0)
    ijk = np.floor((xyz - mn) / size).astype(np.int64)
    key = (ijk[:, 0] * 2_000_003 + ijk[:, 1]) * 2_000_003 + ijk[:, 2]
    del ijk
    _, idx = np.unique(key, return_index=True)
    return xyz[np.sort(idx)]


def read_cloud(path, thin=None):
    """Le LAS/LAZ (laspy), E57 (pacote e57) ou texto XYZ/PTS/CSV (3 primeiras colunas = x y z).
    Retorna array (N,3) float64. thin = tamanho de voxel (m) para reduzir ja na leitura (util em E57 grandes)."""
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    if ext in (".las", ".laz"):
        import laspy
        las = laspy.read(str(p))
        return np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    if ext == ".e57":
        import e57  # pip install e57 (leitor em Rust, sem compilar). Pontos de todos os scans, ja com a pose de cada scan.
        pts = np.asarray(e57.read_points(str(p)).points)[:, :3].astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if thin:
            pts = thin_voxel(pts, thin)
        return pts.astype(np.float64)
    if ext in (".xyz", ".pts", ".txt", ".csv"):
        with open(p, encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            second = f.readline().strip()
        skip = 1 if first.isdigit() else 0  # PTS traz o numero de pontos na primeira linha
        probe = second if skip else first
        sep = "," if "," in probe else (";" if ";" in probe else r"\s+")
        df = pd.read_csv(p, sep=sep, header=None, usecols=[0, 1, 2], skiprows=skip, engine="c" if sep != r"\s+" else "python")
        return df.to_numpy(dtype=np.float64)
    raise ValueError(f"Formato nao suportado: {ext}. Exporte como LAS/LAZ, E57, XYZ ou PTS.")


def write_labeled_las(path, xyz, tree_id, height):
    """Grava LAS/LAZ com tree_id (int32, -1 = sem fuste) e height (altura acima do solo).
    O tree_id pode ser aberto no CloudCompare (campo escalar) ou editado no Segfix."""
    import laspy
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="height", type=np.float32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.tree_id = np.asarray(tree_id, dtype=np.int32)
    las.height = np.asarray(height, dtype=np.float32)
    las.write(str(path))
