"""Confere validation.py contra calculo independente (numpy/scipy) com dados SINTETICOS.
Nao usa dados do artigo; confere so as formulas."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from tls_mangue.validation import metrics, main

rng = np.random.default_rng(0)
f = rng.uniform(5, 15, 8); t = f * 1.05 + rng.normal(0, 0.4, 8)
m = metrics(f, t)
assert abs(m["MAPE_%"] - 100 * np.mean(np.abs(t - f) / f)) < 1e-9
assert abs(m["RMSE"] - np.sqrt(np.mean((t - f) ** 2))) < 1e-9
assert abs(m["RMSE_%"] - 100 * m["RMSE"] / f.mean()) < 1e-9
assert abs(m["r"] - np.corrcoef(f, t)[0, 1]) < 1e-9 and abs(m["R2"] - stats.pearsonr(f, t)[0] ** 2) < 1e-9
ex = np.zeros(8, bool); ex[0] = True
m2 = metrics(f, t, ex)
assert abs(m2["r"] - np.corrcoef(f[1:], t[1:])[0, 1]) < 1e-9 and abs(m2["RMSE"] - m["RMSE"]) < 1e-9  # outlier so sai da regressao
d = pathlib.Path(tempfile.mkdtemp())
plots = [f"P{i}" for i in range(1, 7)]
cols = ["dbh_mean_cm", "height_mean_m", "density_ha", "basal_area_m2ha"]
fld = pd.DataFrame({"plot": plots, **{c: rng.uniform(5, 20, 6) for c in cols}})
fld.to_csv(d / "campo.csv", index=False)
for nm, k in (("A", 1.03), ("B", 1.15)):
    tl = fld.copy(); tl[cols] = tl[cols] * k + rng.normal(0, 0.2, (6, 4)); tl["subset"] = "all"
    tl.to_csv(d / f"{nm}.csv", index=False)
main(["--field", str(d / "campo.csv"), "--tls", str(d / "A.csv"), str(d / "B.csv"), "--names", "A,B",
      "--out", str(d / "val"), "--outliers", "P3:density_ha", "--time-min", "30,10"])
assert (d / "val/metricas.csv").exists() and (d / "val/grafico_1a1_A.png").exists() and (d / "val/custo_beneficio.csv").exists()
print("validation.py OK (formulas conferidas contra numpy/scipy)")
