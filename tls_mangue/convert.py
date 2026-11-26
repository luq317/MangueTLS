"""Converte nuvem E57 (ou LAS, XYZ, PTS) em LAZ para ferramentas que nao leem E57 (Sylva, 3DFin, TLS2trees, FSCT, TreeLS).

Uso:
  python -m tls_mangue.convert --input "F2 2023.e57" --output F2_2023_2cm.laz --voxel 0.02

O LAZ guarda as coordenadas com offset no cabecalho (precisao de 1 mm), entao nao ha perda por float32.
--voxel reduz a nuvem (1 ponto por voxel); 0.02 m e um bom ponto de partida para parcelas de 200 a 300 m2.
--recentralizar subtrai o minimo de x,y,z (coordenadas locais pequenas; use se a ferramenta de destino
perder precisao com UTM).
"""
import argparse
import numpy as np
from .io_clouds import read_cloud, thin_voxel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="arquivo .laz ou .las")
    ap.add_argument("--voxel", type=float, default=0.02, help="m; 0 = sem reducao")
    ap.add_argument("--recentralizar", action="store_true")
    a = ap.parse_args(argv)
    import laspy
    xyz = read_cloud(a.input, thin=a.voxel or None)
    if a.voxel and not a.input.lower().endswith(".e57"):
        xyz = thin_voxel(xyz, a.voxel)
    off = xyz.min(axis=0)
    orig_min = xyz.min(axis=0)
    if a.recentralizar:
        xyz = xyz - orig_min
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(a.output)
    print(f"{len(xyz):,} pontos gravados em {a.output}")
    if a.recentralizar:
        print("minimo original removido (x, y, z):", orig_min.tolist(), "(guarde para voltar ao sistema da nuvem)")


if __name__ == "__main__":
    main()
