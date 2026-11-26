"""Execucao ponta a ponta do Pipeline C. Gera tambem o relatorio em PDF (relatorio_<plot-id>.pdf).

Uso:
  python -m tls_mangue.pipeline --input F1.laz --out saida_F1 --plot-id F1 --plot-area 240
  python -m tls_mangue.pipeline --input F1.laz --out saida_F1 --plot-id F1 --plot-polygon F1_poligono.csv --config params.json

Saidas (pasta --out):
  stems.csv                tabela por fuste (ID, X, Y, DAP, altura, area basal, qualidade, flags)
  plot_summary.csv         agregacao por parcela (todos os fustes e so os de qualidade >= min_quality)
  review_list.csv          fustes para conferir no CloudCompare ou Segfix
  stems_labeled.laz        nuvem com tree_id (int32) e height, para revisao
  plan_view.png            mapa dos fustes com circulos
  run.json                 parametros, versoes e relatorio do processamento
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import __version__
from .config import Params
from .ground import build_dtm, normalize_height
from .io_clouds import read_cloud, write_labeled_las
from .plot_metrics import aggregate, hull_area, inside, polygon_area
from .preprocess import preprocess
from .stems import detect_stems, label_points


def run(input_path, out_dir, plot_id="plot", plot_area=None, polygon=None, params=None, seed=0, write_cloud=True, verbose=True, relatorio=True, comando=None):
    t0 = time.time()
    p = params or Params()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    log(f"[1/7] Lendo {input_path}")
    xyz_raw = read_cloud(input_path, thin=p.voxel_size)
    origin = xyz_raw.min(axis=0)
    xyz = xyz_raw - origin
    log(f"[2/7] Pre-processamento ({len(xyz):,} pontos)")
    xyz, rep_pre = preprocess(xyz, p)
    log(f"      {rep_pre}")
    log("[3/7] Solo e normalizacao de altura")
    dtm, rep_g, _ = build_dtm(xyz, p)
    zn, keep = normalize_height(xyz, dtm, p)
    xyz, zn = xyz[keep], zn[keep]
    xyz_n = np.column_stack([xyz[:, 0], xyz[:, 1], zn])
    log(f"      {rep_g}")
    log("[4/7] Fustes em varias fatias")
    stems, rep_s = detect_stems(xyz_n, p, seed)
    log(f"      {rep_s}")

    poly = None
    poly_utm = None
    area_origem = ""
    if polygon is not None:
        poly_utm = pd.read_csv(polygon)[["x", "y"]].to_numpy(float)
        poly = poly_utm - origin[:2]
        area = polygon_area(poly)
        area_origem = "polígono informado"
    elif plot_area is not None:
        area = float(plot_area)
        area_origem = "área nominal informada"
    else:
        area_origem = "fecho convexo da nuvem"

        area = hull_area(xyz[:, :2])
        log(f"      AVISO: sem --plot-area nem --plot-polygon; usando a area do fecho convexo ({area:.1f} m2).")
    df = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in stems])
    if len(df) == 0:
        raise RuntimeError("Nenhum fuste detectado. Revise slice_min, ransac_tolerance, min_arc_deg e a nuvem.")
    if poly is not None:
        df = df[inside(poly, df["x"], df["y"])].reset_index(drop=True)
        stems = [s for s in stems if s["stem_id"] in set(df["stem_id"])]
    df.insert(0, "plot", plot_id)
    df["x_utm"] = df["x"] + origin[0]
    df["y_utm"] = df["y"] + origin[1]

    log("[5/7] Agregacao por parcela e lista de revisao")
    summ = pd.DataFrame([
        {"plot": plot_id, **aggregate(df, area, 0.0, p.min_dbh_cm, "all")},
        {"plot": plot_id, **aggregate(df, area, p.min_quality, p.min_dbh_cm, f"quality>={p.min_quality}")},
    ])
    review = df[df["flags"].str.len() > 0].sort_values("quality")
    df.to_csv(out / "stems.csv", index=False, float_format="%.4f")
    summ.to_csv(out / "plot_summary.csv", index=False, float_format="%.4f")
    review.to_csv(out / "review_list.csv", index=False, float_format="%.4f")

    log("[6/7] Rotulagem de pontos e figura")
    if write_cloud:
        lab = label_points(xyz_n, stems, p)
        write_labeled_las(out / "stems_labeled.laz", xyz + origin, lab, zn)
    _plan_view(df, xyz_n, poly, out / "plan_view.png", plot_id)
    meta = {
        "tls_mangue_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "input": str(input_path), "plot_id": plot_id, "plot_area_m2": area, "origin_xyz": origin.tolist(),
        "preprocess": rep_pre, "ground": rep_g, "stems": rep_s, "n_stems_in_plot": int(len(df)),
        "n_review": int(len(review)), "seed": seed, "seconds": round(time.time() - t0, 1), "params": p.__dict__,
    }
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if relatorio:
        try:
            from . import relatorio as rel
            log("[7/7] Relatorio em PDF")
            rel.salvar_dados(out, xyz, zn, origin)
            pdfp = rel.gerar(out, plot_id, df, poly=poly_utm, params=p.__dict__, area=area, area_origem=area_origem, versao=__version__, comando=comando)
            log(f"      {pdfp}")
        except Exception as e:  # o relatorio nunca deve derrubar o processamento
            log(f"      AVISO: relatorio nao gerado ({type(e).__name__}: {e}). As tabelas foram salvas normalmente.")
    log(f"Concluido em {meta['seconds']} s. Saidas em {out}")
    return df, summ


def _plan_view(df, xyz_n, poly, path, plot_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    m = (xyz_n[:, 2] > 1.0) & (xyz_n[:, 2] < 1.6)
    ax.scatter(xyz_n[m, 0], xyz_n[m, 1], s=0.3, c="#b8c4cc", linewidths=0)
    sc = ax.scatter(df["x"], df["y"], s=np.maximum(df["dbh_cm"], 2) * 6, c=df["quality"], cmap="viridis", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
    bad = df[df["flags"].str.contains("REVIEW")]
    ax.scatter(bad["x"], bad["y"], s=90, facecolors="none", edgecolors="red", linewidths=1.2, label="revisar")
    if poly is not None:
        ax.plot(*np.vstack([poly, poly[:1]]).T, "k--", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("x (m, local)"); ax.set_ylabel("y (m, local)")
    ax.set_title(f"Parcela {plot_id}: {len(df)} fustes (tamanho = DAP, cor = qualidade)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="qualidade")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="nuvem registrada da parcela (.las, .laz, .xyz, .pts)")
    ap.add_argument("--out", required=True, help="pasta de saida")
    ap.add_argument("--plot-id", default="plot")
    ap.add_argument("--plot-area", type=float, help="area da parcela em m2")
    ap.add_argument("--plot-polygon", help="CSV com colunas x,y dos vertices da parcela (mesmo sistema da nuvem)")
    ap.add_argument("--config", help="JSON com parametros (veja params_exemplo.json)")
    ap.add_argument("--no-adaptive-dbh", action="store_true", help="mede sempre a 1,30 m (para comparar)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem-relatorio", action="store_true", help="nao gera o PDF nem as figuras do relatorio")
    a = ap.parse_args(argv)
    p = Params.from_json(a.config) if a.config else Params()
    if a.no_adaptive_dbh:
        p.adaptive_dbh = False
    run(a.input, a.out, a.plot_id, a.plot_area, a.plot_polygon, p, a.seed, relatorio=not a.sem_relatorio, comando="python -m tls_mangue.pipeline " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
