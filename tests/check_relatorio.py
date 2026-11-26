"""Teste de fumaca do relatorio: roda o pipeline na nuvem sintetica e confere que o PDF e as figuras foram gerados."""
import sys, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tls_mangue.synthetic import make_plot
from tls_mangue.pipeline import run

d = pathlib.Path(tempfile.mkdtemp())
cloud, truth, area = make_plot(seed=1, out_las=d / "s.laz")
df, summ = run(d / "s.laz", d / "out", "SINT", plot_area=area, verbose=True, write_cloud=False)
pdf = d / "out" / "relatorio_SINT.pdf"
figs = sorted(p.name for p in (d / "out" / "figuras").glob("*.png"))
assert pdf.exists() and pdf.stat().st_size > 50_000, "PDF nao gerado"
assert len(figs) >= 12, figs
print("ok:", pdf, pdf.stat().st_size, "bytes;", len(figs), "figuras")
