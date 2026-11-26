"""Etapa 4: validacao TLS x campo (metricas e graficos 1:1 no padrao de Matta et al. 2025).

Metricas (por parametro, n = numero de parcelas):
  MAPE  = mean(|TLS - campo| / campo) * 100
  RMSE  = sqrt(mean((TLS - campo)^2));  RMSE% = 100 * RMSE / mean(campo)
  r     = correlacao de Pearson
  R2    = r^2 da regressao linear TLS (dependente) ~ campo (independente)
Outliers podem ser excluidos apenas das regressoes (r, R2 e reta), como em Matta 2025;
MAPE e RMSE usam todos os pontos. Eles aparecem como quadrados no grafico.

Uso (CLI):
  python -m tls_mangue.validation --field campo.csv --tls plot_summary.csv [--tls outro.csv ...]
        --names SYLVA,3DFIN --out validacao [--outliers P3:dbh_mean_cm,P5:density_ha]
Formato do campo.csv: colunas plot, dbh_mean_cm, height_mean_m, density_ha, basal_area_m2ha
O plot_summary.csv do pipeline tem varias linhas por parcela (coluna subset): usa-se subset == 'all'
(mude com --subset).
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

PARAMS = {
    "dbh_mean_cm": ("DAP medio (cm)", "cm"),
    "height_mean_m": ("Altura media (m)", "m"),
    "density_ha": ("Densidade (troncos/ha)", "troncos/ha"),
    "basal_area_m2ha": ("Area basal (m2/ha)", "m2/ha"),
}


def metrics(field, tls, exclude=None):
    """field, tls: arrays. exclude: mascara booleana de outliers (fora so das regressoes)."""
    f, t = np.asarray(field, float), np.asarray(tls, float)
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    ex = np.zeros(len(f), bool) if exclude is None else np.asarray(exclude, bool)[ok]
    rmse = float(np.sqrt(np.mean((t - f) ** 2)))
    out = {"n": int(len(f)), "n_outliers": int(ex.sum()),
           "MAPE_%": float(np.mean(np.abs(t - f) / f) * 100),
           "RMSE": rmse, "RMSE_%": 100 * rmse / float(np.mean(f)),
           "bias": float(np.mean(t - f))}
    fr, tr = f[~ex], t[~ex]
    if len(fr) >= 3:
        lr = stats.linregress(fr, tr)
        out.update({"r": float(lr.rvalue), "R2": float(lr.rvalue ** 2), "slope": float(lr.slope),
                    "intercept": float(lr.intercept), "p_value": float(lr.pvalue)})
    else:
        out.update({"r": np.nan, "R2": np.nan, "slope": np.nan, "intercept": np.nan, "p_value": np.nan})
    return out


def load_tls(path, subset="all"):
    d = pd.read_csv(path)
    if "subset" in d.columns:
        d = d[d["subset"] == subset]
    return d.set_index("plot")


def parse_outliers(s):
    res = {}
    for tok in filter(None, (s or "").split(",")):
        plot, par = tok.split(":")
        res.setdefault(par.strip(), set()).add(plot.strip())
    return res


def validate(field_df, tls_tables, names, outliers=None, params=PARAMS):
    """Retorna tabela de metricas (ferramenta x parametro) e dicionario de dados pareados."""
    fd = field_df.set_index("plot")
    rows, paired = [], {}
    outliers = outliers or {}
    for nm, tl in zip(names, tls_tables):
        common = fd.index.intersection(tl.index)
        if len(common) == 0:
            raise ValueError(f"Nenhuma parcela em comum entre o campo e '{nm}'")
        for par in params:
            if par not in fd.columns or par not in tl.columns:
                continue
            f, t = fd.loc[common, par].values, tl.loc[common, par].values
            ex = np.array([p in outliers.get(par, set()) for p in common])
            m = metrics(f, t, ex)
            rows.append({"tool": nm, "param": par, **m})
            paired[(nm, par)] = (list(common), f, t, ex)
    return pd.DataFrame(rows), paired


def plot_1to1(paired, names, out_png, params=PARAMS):
    """Painel 1:1 (um por parametro; uma figura por ferramenta). Linha tracejada = 1:1; vermelha = ajuste."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outs = []
    for nm in names:
        pars = [p for p in params if (nm, p) in paired]
        if not pars:
            continue
        fig, axs = plt.subplots(1, len(pars), figsize=(4.0 * len(pars), 4.0), squeeze=False)
        for ax, par in zip(axs[0], pars):
            plots, f, t, ex = paired[(nm, par)]
            m = metrics(f, t, ex)
            ax.scatter(f[~ex], t[~ex], s=36, c="k", zorder=3)
            if ex.any():
                ax.scatter(f[ex], t[ex], s=40, marker="s", c="0.5", zorder=3)
            lo = min(f.min(), t.min()); hi = max(f.max(), t.max()); pad = 0.08 * (hi - lo or 1)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, "--", c="0.4", lw=1)
            if np.isfinite(m["slope"]):
                xs = np.array(lim)
                ax.plot(xs, m["intercept"] + m["slope"] * xs, c="red", lw=1.4)
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
            ax.set_title(PARAMS[par][0], fontsize=10)
            ax.set_xlabel(f"Campo ({PARAMS[par][1]})"); ax.set_ylabel(f"TLS {nm} ({PARAMS[par][1]})")
            ax.text(0.04, 0.96, f"R2={m['R2']:.2f}  r={m['r']:.2f}\nMAPE={m['MAPE_%']:.1f}%\n"
                    f"RMSE={m['RMSE']:.2f} ({m['RMSE_%']:.1f}%)", transform=ax.transAxes, va="top", fontsize=8)
        fig.tight_layout()
        p = pathlib.Path(out_png).with_name(pathlib.Path(out_png).stem + f"_{nm}.png")
        fig.savefig(p, dpi=200); plt.close(fig); outs.append(p)
    return outs


def match_stems(tls, field, tol=0.35):
    """Pareamento fuste a fuste (Hungarian no plano xy; ambos com colunas x, y, dbh_cm).
    Retorna (pares, completude, corretude). Util quando ha coordenadas dos fustes no campo."""
    c = np.hypot(tls.x.values[:, None] - field.x.values[None], tls.y.values[:, None] - field.y.values[None])
    c[c > tol] = 1e6
    r, k = linear_sum_assignment(c)
    ok = c[r, k] < 1e5
    pairs = pd.DataFrame({"tls_idx": r[ok], "field_idx": k[ok],
                          "dbh_tls": tls.dbh_cm.values[r[ok]], "dbh_field": field.dbh_cm.values[k[ok]]})
    return pairs, ok.sum() / len(field), ok.sum() / len(tls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True); ap.add_argument("--tls", required=True, nargs="+")
    ap.add_argument("--names"); ap.add_argument("--out", default="validacao")
    ap.add_argument("--subset", default="all"); ap.add_argument("--outliers", default="")
    ap.add_argument("--time-min", help="tempo de processamento (min) por ferramenta, separado por virgula")
    a = ap.parse_args(argv)
    names = a.names.split(",") if a.names else [pathlib.Path(p).stem for p in a.tls]
    if len(names) != len(a.tls):
        raise SystemExit("--names precisa ter o mesmo numero de itens que --tls")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, paired = validate(pd.read_csv(a.field), [load_tls(p, a.subset) for p in a.tls], names, parse_outliers(a.outliers))
    res.round(4).to_csv(out / "metricas.csv", index=False)
    pngs = plot_1to1(paired, names, out / "grafico_1a1.png")
    if a.time_min:
        tm = dict(zip(names, map(float, a.time_min.split(","))))
        s = res.pivot(index="tool", columns="param", values="MAPE_%")
        s["tempo_min"] = pd.Series(tm); s.round(2).to_csv(out / "custo_beneficio.csv")
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string(index=False))
    print("salvo em", out.resolve(), [p.name for p in pngs])


if __name__ == "__main__":
    main()
