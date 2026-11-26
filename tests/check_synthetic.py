"""Teste de fumaça: roda o pipeline na nuvem sintetica e compara com a verdade conhecida.
Os numeros aqui verificam o CODIGO, nao a ciencia."""
import sys, time, tempfile, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scipy.optimize import linear_sum_assignment
from tls_mangue.synthetic import make_plot
from tls_mangue.config import Params
from tls_mangue.pipeline import run


def match(df, truth, tol=0.35):
    c = np.hypot(df.x.values[:, None] - truth.x.values[None], df.y.values[:, None] - truth.y.values[None])
    c[c > tol] = 1e6
    r, k = linear_sum_assignment(c)
    ok = c[r, k] < 1e5
    return df.iloc[r[ok]].reset_index(drop=True), truth.iloc[k[ok]].reset_index(drop=True)


def report(tag, df, truth):
    m, t = match(df, truth)
    err = m.dbh_cm - t.dbh_cm
    print(f"[{tag}] detectados={len(df)} verdade={len(truth)} pareados={len(m)} "
          f"completude={len(m)/len(truth):.2%} corretude={len(m)/len(df):.2%}")
    print(f"      DAP: vies={err.mean():+.2f} cm, RMSE={np.sqrt((err**2).mean()):.2f} cm ({100*np.sqrt((err**2).mean())/t.dbh_cm.mean():.1f}%), MAPE={np.mean(np.abs(err)/t.dbh_cm)*100:.1f}%")
    root = t.h_root_m.values > 1.3
    for nm, sel in (("raiz acima de 1,3 m", root), ("demais", ~root)):
        if sel.sum():
            e = err.values[sel]
            print(f"      {nm}: n={sel.sum()}, vies={e.mean():+.2f} cm, RMSE={np.sqrt((e**2).mean()):.2f} cm")
    he = (m.height_m - t.height_m).dropna()
    print(f"      Altura: vies={he.mean():+.2f} m, RMSE={np.sqrt((he**2).mean()):.2f} m")
    return m, t


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d = pathlib.Path(tempfile.mkdtemp())
    cloud, truth, area = make_plot(seed=seed, out_las=d / "sint.laz")
    print(f"nuvem sintetica: {len(cloud):,} pontos, {len(truth)} fustes, {int((truth.h_root_m>1.3).sum())} com raiz acima de 1,3 m")
    for tag, adaptive in (("DAP adaptativo", True), ("DAP fixo 1,30 m", False)):
        p = Params(); p.adaptive_dbh = adaptive
        t0 = time.time()
        df, summ = run(d / "sint.laz", d / ("out_" + str(adaptive)), "SINT", plot_area=area, params=p, verbose=False, write_cloud=(adaptive), relatorio=False)
        print(f"{tag}: {time.time()-t0:.1f} s")
        report(tag, df, truth)
    print(summ.to_string())
    print("saidas em", d)
