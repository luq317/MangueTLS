"""Avalia o pipeline (ou outra ferramenta) numa parcela do NEMA com os tres arquivos do laboratorio:
nuvem E57, CSV de anotacoes exportado do ReCap (x,y,z,rotulo,obs) e planilha de extracao manual (aba X-L).

Formato dos rotulos do CSV: "Rh 12 viva", "Av 8 morta", "Lg 14 viva" (numero = coluna Individuo da planilha);
"Cano 1..4" = cantos da parcela (viram o poligono); outros rotulos sao ignorados.

Uso:
  python -m tls_mangue.avalia_parcela --nome H2 --e57 H2_2023.e57 --csv H2_2023.csv --xlsx H2_2023.xlsx --aba H2-L \
        --out saida_H2 --config params.json
  python -m tls_mangue.avalia_parcela --nome H2 --csv ... --xlsx ... --aba H2-L --out saida --stems-csv sylva.csv --dbh-em-m
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def ler_planilha(xlsx, aba):
    """Le a aba de uma parcela (colunas: Individuo, Nº tr's, Obs, Lacre, Especie, DAP, ..., Alt. Corr., Condicao).
    Mantem toda linha com DAP numerico e especie valida, mesmo sem numero de Individuo (ex. '42B' ou celula vazia);
    essas linhas recebem ind negativo unico e nao casam com rotulos do ReCap, mas entram nos totais da parcela.
    Para na linha 3000 para nao ler a tabelinha auxiliar de classes que vem depois da tabela principal."""
    raw = pd.read_excel(xlsx, sheet_name=aba, header=None)
    area = float(raw.iloc[3, 1])
    d = raw.iloc[7:3000, :14].copy()
    d.columns = ["ind", "ntr", "obs", "lacre", "sp", "dap", "tel", "vara", "regua", "alt", "altc", "cond", "flag", "ab"]
    d["sp"] = d.sp.astype(str).str.lower().str.strip().str[:2].replace({"rg": "rh"})
    d["dap"] = pd.to_numeric(d.dap, errors="coerce")
    d["altc"] = pd.to_numeric(d.altc, errors="coerce")
    d["cond"] = d.cond.astype(str).str.lower().str.strip().str[:1]
    d = d[d.dap.notna() & d.sp.isin(["av", "lg", "rh"])].copy()
    ind = pd.to_numeric(d.ind, errors="coerce")
    d["ind"] = [int(v) if pd.notna(v) else -(k + 1) for k, v in enumerate(ind)]
    return d.reset_index(drop=True), area


def ler_anotacoes(csv):
    c = pd.read_csv(csv, header=None, names=["x", "y", "z", "lab", "obs"])
    corners, rows = [], []
    for r in c.itertuples():
        lab = str(r.lab)
        if re.match(r"\s*cano\s*\d+", lab, re.I):
            corners.append((r.x, r.y)); continue
        m = re.match(r"\s*(rh|av|lg)\s*(\d+)\s*(viva|morta|v|m)?", lab, re.I)
        if m:
            cond = (m.group(3) or "").lower()[:1]
            rows.append(dict(x=r.x, y=r.y, z=r.z, lab=lab, sp=m.group(1).lower(), ind=int(m.group(2)), cond=cond))
    ann = pd.DataFrame(rows)
    poly = None
    if len(corners) >= 3:
        P = np.array(corners); ctr = P.mean(0)
        P = P[np.argsort(np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0]))]
        poly = P
    return ann, poly


def area_poligono(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def metricas(e, ref):
    e = np.asarray(e, float); ref = np.asarray(ref, float)
    if len(e) == 0:
        return {}
    pred = ref + e
    out = {"n": int(len(e)), "mediana_erro_abs": float(np.median(np.abs(e))), "dentro_de_1": float(np.mean(np.abs(e) <= 1.0)), "vies": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "rmse_pct": float(100 * np.sqrt((e ** 2).mean()) / ref.mean()), "mape_pct": float(100 * np.mean(np.abs(e) / ref))}
    if len(e) > 2 and np.std(pred) > 0 and np.std(ref) > 0:
        r = float(np.corrcoef(pred, ref)[0, 1]); out["r"] = r; out["r2"] = r * r
    return out


def avaliar(stems, ann, sheet, area_nominal, poly, nome, out, tol=0.4, poly_origem=""):
    """stems: DataFrame com x_utm, y_utm, dbh_cm, height_m (ja recortado pelo poligono). Retorna dict de resultados."""
    os.makedirs(out, exist_ok=True)
    M = np.hypot(stems.x_utm.values[:, None] - ann.x.values[None], stems.y_utm.values[:, None] - ann.y.values[None])
    M[M > tol] = 1e6
    r, k = linear_sum_assignment(M); ok = M[r, k] < 1e5; r, k = r[ok], k[ok]
    key = {(row.sp, row.ind): row for row in sheet.itertuples()}
    rows = []
    for i, j in zip(r, k):
        a = ann.iloc[j]; s = key.get((a.sp, a.ind))
        rows.append(dict(stem_idx=i, x_pipe=stems.x_utm.iloc[i], y_pipe=stems.y_utm.iloc[i], rotulo=a.lab, cond_ann=a.cond, dist_m=M[i, j], dbh_pipe=stems.dbh_cm.iloc[i], h_pipe=stems.height_m.iloc[i] if "height_m" in stems else np.nan,
                         dap_man=s.dap if s else np.nan, h_man=s.altc if s else np.nan, cond_man=s.cond if s else ""))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(out, "pareamento_por_fuste.csv"), index=False)
    live_ann = ann[ann.cond.isin(["v", ""])]
    in_poly = np.ones(len(ann), bool)
    if poly is not None:
        from matplotlib.path import Path
        in_poly = Path(poly).contains_points(ann[["x", "y"]].to_numpy())
    live_in = ann[(ann.cond == "v") & in_poly]
    matched_live = P[P.cond_ann == "v"] if len(P) else P
    res = {"parcela": nome, "poligono_origem": poly_origem, "area_poligono_m2": area_poligono(poly) if poly is not None else np.nan, "area_planilha_m2": area_nominal,
           "anot_vivas_no_poligono": int(len(live_in)), "anot_vivas_detectadas": int(len(matched_live)),
           "recall_vivas": float(len(matched_live) / max(len(live_in), 1)), "fustes_pipeline_no_poligono": int(len(stems)),
           "individuos_planilha": int(len(sheet)), "planilha_vivos": int((sheet.cond == "v").sum()), "planilha_mortos": int((sheet.cond == "m").sum())}
    V = P[(P.cond_ann == "v") & (P.cond_man == "v") & P.dap_man.notna()] if len(P) else P
    if len(V):
        res["dap"] = metricas(V.dbh_pipe - V.dap_man, V.dap_man)
        fin = V[V.dap_man < 5]; gro = V[V.dap_man >= 5]
        res["dap_finos_<5cm"] = {"n": int(len(fin)), "vies": float((fin.dbh_pipe - fin.dap_man).mean()) if len(fin) else None}
        res["dap_>=5cm"] = {"n": int(len(gro)), "vies": float((gro.dbh_pipe - gro.dap_man).mean()) if len(gro) else None}
        H = V[V.h_man.notna() & V.h_pipe.notna()]
        if len(H):
            res["altura"] = metricas(H.h_pipe - H.h_man, H.h_man)
    # totais da parcela (area do poligono se houver)
    A = res["area_poligono_m2"] if np.isfinite(res["area_poligono_m2"]) else area_nominal
    v = sheet[sheet.cond == "v"]
    res["totais"] = {
        "area_usada_m2": A,
        "manual_vivos": {"n": int(len(v)), "dap_medio": float(v.dap.mean()), "altura_media": float(v.altc.mean()), "densidade_ha": float(len(v) / A * 1e4),
                         "area_basal_m2ha": float((np.pi * (v.dap / 200) ** 2).sum() / A * 1e4)},
        "pipeline_todos_no_poligono": {"n": int(len(stems)), "dap_medio": float(stems.dbh_cm.mean()) if len(stems) else None,
                                       "altura_media": float(stems.height_m.mean()) if "height_m" in stems and len(stems) else None,
                                       "densidade_ha": float(len(stems) / A * 1e4), "area_basal_m2ha": float((np.pi * (stems.dbh_cm / 200) ** 2).sum() / A * 1e4)}}
    if "flags" in stems:
        ok_r = ~stems["flags"].fillna("").str.contains("REVIEW")
        t = stems[ok_r]
        res["totais"]["pipeline_sem_REVIEW"] = {"n": int(len(t)), "dap_medio": float(t.dbh_cm.mean()) if len(t) else None,
            "densidade_ha": float(len(t) / A * 1e4), "area_basal_m2ha": float((np.pi * (t.dbh_cm / 200) ** 2).sum() / A * 1e4)}
    with open(os.path.join(out, "avaliacao.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    _figuras(stems, ann, poly, P, V, nome, out)
    return res


def _figuras(stems, ann, poly, P, V, nome, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    NAVY, TEAL, GRAY = "#1F3B57", "#2A9D8F", "#9AA5B1"
    fig, ax = plt.subplots(figsize=(7, 7))
    if poly is not None:
        ax.plot(*np.vstack([poly, poly[:1]]).T, "-", color=NAVY, lw=1)
    ax.scatter(stems.x_utm, stems.y_utm, s=np.maximum(stems.dbh_cm, 1) * 8, facecolors="none", edgecolors=TEAL, lw=1.2, label="pipeline (tamanho = DAP)")
    for c, m, lab in (("v", "+", "anotacao viva"), ("m", "x", "anotacao morta")):
        a = ann[ann.cond == c]; ax.scatter(a.x, a.y, marker=m, s=40, color=NAVY if c == "v" else "#C0392B", label=lab)
    ax.set_aspect("equal"); ax.set_title(f"{nome}: fustes do pipeline e anotacoes do ReCap"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, "planta.png"), dpi=150); plt.close(fig)
    if len(V) > 2:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
        for ax, (a, b, t) in zip(axs, ((V.dap_man, V.dbh_pipe, "DAP (cm)"), (V.h_man, V.h_pipe, "Altura (m)"))):
            ok = a.notna() & b.notna()
            if ok.sum() == 0:
                ax.set_title(t + ' (sem dados)'); continue
            ax.scatter(a[ok], b[ok], s=22, color=TEAL)
            lim = [0, max(a[ok].max(), b[ok].max()) * 1.05]; ax.plot(lim, lim, "--", color=GRAY); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("TLS manual (ReCap)"); ax.set_ylabel("Pipeline"); ax.set_title(t)
        fig.suptitle(f"{nome}: 1:1 (vivos pareados)"); fig.tight_layout(); fig.savefig(os.path.join(out, "1a1.png"), dpi=150); plt.close(fig)


def poligono_das_anotacoes(ann, buffer=0.3):
    """Retangulo de area minima das arvores anotadas, ampliado por buffer (m). Usado quando o CSV nao tem os cantos."""
    from scipy.spatial import ConvexHull
    pts = ann[["x", "y"]].to_numpy()
    hull = pts[ConvexHull(pts).vertices]
    best = None
    for k in range(len(hull)):
        e = hull[(k + 1) % len(hull)] - hull[k]
        ang = np.arctan2(e[1], e[0])
        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        q = hull @ R.T
        a = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or a < best[0]:
            best = (a, R, q)
    _, R, q = best
    x0, x1, y0, y1 = q[:, 0].min() - buffer, q[:, 0].max() + buffer, q[:, 1].min() - buffer, q[:, 1].max() + buffer
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True); ap.add_argument("--csv", required=True); ap.add_argument("--xlsx", required=True); ap.add_argument("--aba", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--e57"); ap.add_argument("--config"); ap.add_argument("--stems-csv"); ap.add_argument("--dbh-em-m", action="store_true")
    ap.add_argument("--tol", type=float, default=0.4)
    ap.add_argument("--dados-dir", help="pasta com relatorio_dados.npz (para o relatorio ao avaliar uma tabela externa)")
    ap.add_argument("--sem-relatorio", action="store_true")
    a = ap.parse_args(argv)
    sheet, area_nom = ler_planilha(a.xlsx, a.aba)
    ann, poly = ler_anotacoes(a.csv)
    os.makedirs(a.out, exist_ok=True)
    poly_origem = "cantos (pontos Cano) do CSV"
    if poly is None:
        poly = poligono_das_anotacoes(ann)
        poly_origem = "retângulo mínimo das árvores anotadas + 0,3 m (o CSV não tem os cantos)"
    pd.DataFrame(poly, columns=["x", "y"]).to_csv(os.path.join(a.out, "poligono_cantos.csv"), index=False)
    pc = os.path.join(a.out, "poligono_cantos.csv")
    dados = a.dados_dir
    if a.stems_csv:
        s = pd.read_csv(a.stems_csv)
        if a.dbh_em_m: s["dbh_cm"] = s["dbh"] * 100
        if "x_utm" not in s: s["x_utm"], s["y_utm"] = s["x"], s["y"]
        from matplotlib.path import Path
        s = s[Path(poly).contains_points(s[["x_utm", "y_utm"]].to_numpy())].reset_index(drop=True)
        if dados is None and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(a.stems_csv)), "relatorio_dados.npz")):
            dados = os.path.dirname(os.path.abspath(a.stems_csv))
    else:
        from .config import Params
        from .pipeline import run
        p = Params.from_json(a.config) if a.config else Params()
        s, _ = run(a.e57, os.path.join(a.out, "pipeline"), a.nome, polygon=pc, params=p, verbose=False, write_cloud=False)
        dados = os.path.join(a.out, "pipeline")
    res = avaliar(s, ann, sheet, area_nom, poly, a.nome, a.out, a.tol, poly_origem)
    if dados and not a.sem_relatorio and os.path.exists(os.path.join(dados, "relatorio_dados.npz")):
        try:
            from . import relatorio as rel
            from . import __version__
            pr = {}
            rj = os.path.join(dados, "run.json")
            if os.path.exists(rj):
                pr = json.load(open(rj, encoding="utf-8")).get("params", {})
            Pm = pd.read_csv(os.path.join(a.out, "pareamento_por_fuste.csv"))
            txt = (f"Polígono: {poly_origem}. Pareamento por posição (tolerância {a.tol} m) e por rótulo (espécie e número do indivíduo) com a planilha {os.path.basename(a.xlsx)}, aba {a.aba}. "
                   "Densidade e área basal do pipeline contam todos os fustes (vivos e mortos); os do manual contam só vivos.")
            pdfp = rel.gerar(dados, a.nome, s, poly=poly, params=pr, area=res["area_poligono_m2"], area_origem=poly_origem, versao=__version__, ref=dict(ann=ann, pareamento=Pm),
                             avaliacao=res, ref_texto=txt, comando="python -m tls_mangue.avalia_parcela " + " ".join(sys.argv[1:]) if False else None)
            import shutil
            shutil.copy(pdfp, os.path.join(a.out, os.path.basename(str(pdfp))))
            print("relatorio:", os.path.join(a.out, os.path.basename(str(pdfp))), file=sys.stderr)
        except Exception as e:
            print(f"AVISO: relatorio com comparacao nao gerado ({type(e).__name__}: {e})", file=sys.stderr)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
