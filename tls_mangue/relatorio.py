"""Relatorio automatico em PDF por parcela (figuras da nuvem, das arvores detectadas e, se houver referencia,
da comparacao com o TLS manual do ReCap).

O pipeline chama `gerar()` ao final de cada execucao (desligue com --sem-relatorio). O avaliador de parcelas
(`avalia_parcela`) chama de novo com a referencia para acrescentar a comparacao.

Arquivos gerados na pasta de saida:
  relatorio_dados.npz   amostra da nuvem + grades (MDT, densidade, altura do dossel) usadas nas figuras
  figuras/*.png         todas as figuras, uma por arquivo
  relatorio_<nome>.pdf  o relatorio

Todos os textos interpretativos sao fixos e gerais. Nada aqui descreve o resultado de uma parcela especifica.
"""
import json
import pathlib

import numpy as np
import pandas as pd

NAVY, TEAL, GRAY, RED, ORG = "#1F3B57", "#2A9D8F", "#9AA5AE", "#C0392B", "#E08E0B"
FLAGS_DOC = {
    "ROOT_ADJUSTED": "DAP medido acima de 1,35 m porque o raio na janela de 1,3 m estava alargado (raiz-escora).",
    "NO_SLICE_AT_DBH": "nenhuma fatia perto da altura de medição; usada a fatia mais próxima.",
    "DBH_EXTRAPOLATED": "a fatia mais próxima está a mais de max_dbh_offset da altura de medição.",
    "HIGH_DBH_HEIGHT": "altura de medição acima de max_dbh_height; provável medida em outra estrutura.",
    "LOW_ARC": "arco visto do círculo menor que 180 graus (fuste visto de um lado só).",
    "FEW_SLICES": "menos de 6 fatias na cadeia do fuste.",
    "UNSTABLE_RADIUS": "raio varia mais que stable_cv na janela do DAP.",
    "REVIEW": "qualidade baixa ou combinação de flags; conferir no CloudCompare ou Segfix.",
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ------------------------------------------------------------------ dados
def salvar_dados(out, xyz_loc, H, origin, max_low=1_200_000, max_high=400_000, seed=0):
    """xyz_loc: pontos em coordenadas locais (origem subtraida), H: altura acima do solo de cada ponto."""
    from scipy.stats import binned_statistic_2d
    out = pathlib.Path(out)
    x = xyz_loc[:, 0] + origin[0]
    y = xyz_loc[:, 1] + origin[1]
    zt = xyz_loc[:, 2] + origin[2] - H
    xe = np.arange(np.percentile(x, 0.2), np.percentile(x, 99.8) + 0.5, 0.5)
    ye = np.arange(np.percentile(y, 0.2), np.percentile(y, 99.8) + 0.5, 0.5)
    mdt = binned_statistic_2d(x, y, zt, "mean", bins=[xe, ye]).statistic
    cnt = binned_statistic_2d(x, y, zt, "count", bins=[xe, ye]).statistic
    chm = binned_statistic_2d(x, y, H, "max", bins=[xe, ye]).statistic
    pb = np.arange(-0.5, max(float(np.nanmax(H)), 1.0) + 0.5, 0.25)
    pc, _ = np.histogram(H, bins=pb)
    rng = np.random.default_rng(seed)
    low = np.where(H <= 4.5)[0]
    high = np.where(H > 4.5)[0]
    il = rng.choice(low, min(len(low), max_low), replace=False) if len(low) else low
    ih = rng.choice(high, min(len(high), max_high), replace=False) if len(high) else high
    idx = np.sort(np.concatenate([il, ih]))
    pts = np.column_stack([xyz_loc[idx, 0], xyz_loc[idx, 1], H[idx]]).astype(np.float32)  # x,y locais (float32 sem perda)
    cortes = {}
    for k, hh in enumerate((0.5, 1.3, 2.5, 4.0)):
        m = np.where(np.abs(H - hh) < 0.06)[0]
        if len(m) > 250_000:
            m = rng.choice(m, 250_000, replace=False)
        cortes[f"corte_{k}"] = np.column_stack([xyz_loc[m, 0], xyz_loc[m, 1]]).astype(np.float32)
    np.savez_compressed(out / "relatorio_dados.npz", pts=pts, origin=np.asarray(origin, float), xe=xe, ye=ye, mdt=mdt, cnt=cnt, chm=chm,
                        perf_bins=pb, perf_cnt=pc, n_pontos=len(x), cortes_h=np.array([0.5, 1.3, 2.5, 4.0]), **cortes)


# ------------------------------------------------------------------ figuras
class _Ctx:
    pass


def _carregar(out, poly, stems):
    z = np.load(pathlib.Path(out) / "relatorio_dados.npz")
    c = _Ctx()
    c.o = z["origin"]
    c.X = z["pts"][:, :2].astype(float) + c.o[:2]
    c.H = z["pts"][:, 2].astype(float)
    c.z = z
    c.poly = None if poly is None else np.asarray(poly, float)
    xs = [c.X[:, 0], stems.x_utm.values]
    ys = [c.X[:, 1], stems.y_utm.values]
    if c.poly is not None:
        xl = (c.poly[:, 0].min() - 2, c.poly[:, 0].max() + 2)
        yl = (c.poly[:, 1].min() - 2, c.poly[:, 1].max() + 2)
    else:
        xl = (np.percentile(c.X[:, 0], 0.3), np.percentile(c.X[:, 0], 99.7))
        yl = (np.percentile(c.X[:, 1], 0.3), np.percentile(c.X[:, 1], 99.7))
    c.xl, c.yl = xl, yl
    return c


def _base(ax, c, title, poly=True):
    ax.set_xlim(*c.xl)
    ax.set_ylim(*c.yl)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9)
    if poly and c.poly is not None:
        ax.plot(*np.vstack([c.poly, c.poly[:1]]).T, "--", c=NAVY, lw=1)


def _save(plt, fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nuvem_fundo(ax, c, alpha=1.0, bins=0.08):
    from matplotlib.colors import LogNorm
    m = (c.H > 0.3) & (c.H < 3)
    nx = max(int(np.ptp(c.xl) / bins), 10)
    ny = max(int(np.ptp(c.yl) / bins), 10)
    return ax.hist2d(c.X[m, 0], c.X[m, 1], bins=[nx, ny], range=[c.xl, c.yl], cmap="Greys", cmin=1, norm=LogNorm(), alpha=alpha)


def figuras(out, stems, poly=None, nome="parcela", ref=None):
    """Gera as figuras. ref (opcional) = dict(ann=DataFrame x,y,lab,cond | pareamento=DataFrame de avalia_parcela)."""
    from matplotlib.patches import Circle, FancyBboxPatch
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    plt = _mpl()
    out = pathlib.Path(out)
    fd = out / "figuras"
    fd.mkdir(exist_ok=True)
    c = _carregar(out, poly, stems)
    s = stems.reset_index(drop=True)
    feitas = []

    def save(fig, name):
        _save(plt, fig, fd / name)
        feitas.append(name)

    # planta da nuvem sem marcacoes
    fig, ax = plt.subplots(figsize=(8, 7.6))
    h2 = _nuvem_fundo(ax, c)
    plt.colorbar(h2[3], ax=ax, shrink=.7, label="pontos por célula de 8 cm (escala log)")
    _base(ax, c, f"{nome}: nuvem em planta (pontos de 0,3 a 3 m acima do solo)", poly=False)
    save(fig, "g01_planta_nuvem.png")

    # vistas laterais
    ii = np.random.default_rng(0).choice(len(c.X), min(len(c.X), 600_000), replace=False)
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, k, lab in ((axs[0], 0, "x (m)"), (axs[1], 1, "y (m)")):
        ax.scatter(c.X[ii, k], c.H[ii], s=0.2, c=c.H[ii], cmap="viridis", rasterized=True)
        ax.set_xlabel(lab)
        ax.set_ylabel("altura acima do solo (m)")
    axs[0].set_title("Vistas laterais (cor = altura acima do solo)", fontsize=9)
    save(fig, "g02_vistas_laterais.png")

    # perfil vertical
    pb, pc = c.z["perf_bins"], c.z["perf_cnt"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8))
    axs[0].barh(pb[:-1], pc, height=0.25, align="edge", color=NAVY)
    axs[0].set_xlabel("pontos por faixa de 25 cm")
    axs[0].set_ylabel("altura acima do solo (m)")
    axs[0].set_title("Perfil vertical de pontos", fontsize=9)
    axs[1].barh(pb[:-1], np.cumsum(pc[::-1])[::-1] / max(pc.sum(), 1) * 100, height=0.25, align="edge", color=TEAL)
    axs[1].set_xlabel("% dos pontos acima da altura")
    axs[1].set_title("Acumulado", fontsize=9)
    save(fig, "g03_perfil_vertical.png")

    # MDT e densidade
    xe, ye = c.z["xe"], c.z["ye"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    im = axs[0].pcolormesh(xe, ye, c.z["mdt"].T, cmap="cividis")
    plt.colorbar(im, ax=axs[0], shrink=.8, label="z do terreno (m)")
    _base(axs[0], c, "Modelo digital do terreno (MDT)")
    cn = np.where(c.z["cnt"] > 0, c.z["cnt"], np.nan)
    im = axs[1].pcolormesh(xe, ye, cn.T, cmap="Greys", norm=LogNorm(vmin=1))
    plt.colorbar(im, ax=axs[1], shrink=.8, label="pontos por célula (log)")
    _base(axs[1], c, "Densidade de pontos por célula de 0,5 m (lacunas = branco)")
    save(fig, "g04_mdt.png")

    # altura do dossel
    fig, ax = plt.subplots(figsize=(8, 7.4))
    im = ax.pcolormesh(xe, ye, c.z["chm"].T, cmap="YlGn", vmin=0, vmax=np.nanpercentile(c.z["chm"], 99))
    plt.colorbar(im, ax=ax, shrink=.75, label="altura máxima (m)")
    _base(ax, c, "Altura máxima dos pontos por célula de 0,5 m")
    save(fig, "g05_altura_dossel.png")

    # cortes horizontais
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for k, ax in enumerate(axs.ravel()):
        P = c.z[f"corte_{k}"].astype(float) + c.o[:2]
        ax.scatter(P[:, 0], P[:, 1], s=0.6, c=GRAY, rasterized=True)
        for r in s.itertuples():
            ax.add_patch(Circle((r.x_utm, r.y_utm), r.dbh_cm / 200, fc="none", ec=TEAL, lw=1.0))
        _base(ax, c, f"Corte a {c.z['cortes_h'][k]:.1f} m (12 cm de espessura); círculos = fustes (diâmetro = DAP)")
        ax.title.set_fontsize(8)
    save(fig, "g06_cortes_horizontais.png")

    # mapas dos fustes
    def tree_map(col, cmap, label, title, name, vmin=None, vmax=None, flag=False):
        fig, ax = plt.subplots(figsize=(8.4, 7.8))
        _nuvem_fundo(ax, c, alpha=.35, bins=0.12)
        sc = ax.scatter(s.x_utm, s.y_utm, c=s[col], cmap=cmap, s=np.clip(s.dbh_cm, 3, 40) ** 1.7 * 1.3, edgecolor="k", lw=.5, vmin=vmin, vmax=vmax, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=.75, label=label)
        if flag:
            f = s[s["flags"].fillna("").str.contains("REVIEW")]
            ax.scatter(f.x_utm, f.y_utm, s=170, facecolor="none", edgecolor=RED, lw=1.2, zorder=4, label="REVIEW")
            ax.legend(loc="upper left", fontsize=8)
        for r in s.itertuples():
            ax.text(r.x_utm + .1, r.y_utm + .1, str(int(r.stem_id)), fontsize=5.5, color=NAVY)
        _base(ax, c, title)
        save(fig, name)
    dmax = max(15, float(np.nanpercentile(s.dbh_cm, 95)))
    hmax = max(6, float(np.nanpercentile(s.height_m.dropna(), 95))) if s.height_m.notna().any() else 10
    tree_map("dbh_cm", "viridis", "DAP (cm)", "Fustes detectados: cor e tamanho = DAP (número = stem_id)", "g07_arvores_dap.png", vmin=0, vmax=dmax)
    tree_map("height_m", "YlGn", "altura (m)", "Fustes detectados: cor = altura", "g08_arvores_altura.png", vmin=0, vmax=hmax)
    tree_map("quality", "RdYlGn", "qualidade do ajuste (0 a 1)", "Fustes detectados: cor = qualidade; anel vermelho = REVIEW", "g09_arvores_qualidade.png", vmin=0.3, vmax=1, flag=True)

    # distribuicoes e flags
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    axs[0].hist(s.dbh_cm, bins=np.arange(0, max(35, s.dbh_cm.max() + 2.5), 2.5), color=TEAL)
    axs[0].set_xlabel("DAP (cm)")
    axs[0].set_ylabel("número de fustes")
    axs[1].hist(s.height_m.dropna(), bins=np.arange(0, max(16, s.height_m.max() + 1), 1), color=NAVY)
    axs[1].set_xlabel("altura (m)")
    axs[2].scatter(s.dbh_cm, s.height_m, c=s.quality, cmap="RdYlGn", vmin=0.3, vmax=1, s=26, edgecolor="k", lw=.3)
    axs[2].set_xlabel("DAP (cm)")
    axs[2].set_ylabel("altura (m)")
    fig.suptitle("Distribuições de DAP e altura dos fustes detectados (cor = qualidade)", fontsize=9)
    save(fig, "g10_distribuicoes.png")

    fl = s["flags"].fillna("").str.split(";").explode()
    fl = fl[fl != ""].value_counts()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(fl):
        axs[0].barh(fl.index[::-1], fl.values[::-1], color=ORG)
    axs[0].set_xlabel("fustes com a flag")
    axs[0].set_title("Flags (um fuste pode ter várias)", fontsize=9)
    axs[1].hist(s.quality, bins=np.linspace(0, 1, 21), color=NAVY)
    axs[1].axvline(0.5, c=RED, ls="--")
    axs[1].set_xlabel("qualidade (linha vermelha = min_quality)")
    save(fig, "g11_flags_qualidade.png")

    # exemplos de fustes
    cand = list(s.sort_values("dbh_cm", ascending=False).index[:1])
    cand += list(s[s["flags"].fillna("").str.contains("ROOT_ADJUSTED")].sort_values("quality", ascending=False).index[:2])
    cand += list(s[s["flags"].isna()].sort_values("quality", ascending=False).index[:2])
    cand += list(s[s["flags"].fillna("").str.contains("REVIEW")].sort_values("quality").index[:1])
    cand = list(dict.fromkeys(cand))
    if cand:
        fig, axs = plt.subplots(1, len(cand), figsize=(2.4 * len(cand), 6), sharey=True)
        for ax, i in zip(np.atleast_1d(axs), cand):
            r = s.loc[i]
            mm = (np.abs(c.X[:, 1] - r.y_utm) < 0.6) & (np.abs(c.X[:, 0] - r.x_utm) < 0.9)
            ax.scatter(c.X[mm, 0] - r.x_utm, c.H[mm], s=1.5, c=GRAY, rasterized=True)
            ax.axhline(r.dbh_height_m, c=TEAL, lw=1.2)
            if np.isfinite(r.height_m):
                ax.axhline(r.height_m, c=RED, lw=1.2, ls="--")
            fg = str(r["flags"]) if isinstance(r["flags"], str) else "sem flags"
            ax.set_title(f"stem {int(r.stem_id)}\nDAP {r.dbh_cm:.1f} cm, h {r.height_m:.1f} m\nq={r.quality:.2f}\n{fg[:38]}", fontsize=6.3)
            ax.set_xlim(-.9, .9)
            ax.set_ylim(0, min(float(c.H.max()), 16))
            ax.set_xlabel("x relativo (m)", fontsize=7)
        np.atleast_1d(axs)[0].set_ylabel("altura acima do solo (m)")
        fig.suptitle("Exemplos de fustes em vista lateral: verde = altura de medição do DAP; vermelho tracejado = topo do fuste", fontsize=8.5)
        save(fig, "g12_exemplos_fustes.png")

    # fluxo
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    steps = [("1. Leitura", "E57/LAS/LAZ/PTS\nvoxel_size"), ("2. Limpeza", "SOR e ROR"), ("3. Solo", "MDT por percentil\nground_*"),
             ("4. Fatias", "círculos por fatia\nslice_*, r_min/r_max,\nransac_tolerance, min_arc_deg"), ("5. Cadeias", "liga fatias, remove raízes\nmin_chain_slices, root_*"),
             ("6. Medidas", "DAP adaptativo, altura,\nqualidade, flags")]
    for k, (t, sub) in enumerate(steps):
        x0 = 0.2 + k * 1.95
        ax.add_patch(FancyBboxPatch((x0, 2.2), 1.7, 1.9, boxstyle="round,pad=0.05", fc="#E8EEF3", ec=NAVY))
        ax.text(x0 + .85, 3.75, t, ha="center", fontsize=9, color=NAVY, weight="bold")
        ax.text(x0 + .85, 2.95, sub, ha="center", va="center", fontsize=6.3)
        if k < 5:
            ax.annotate("", xy=(x0 + 1.93, 3.15), xytext=(x0 + 1.72, 3.15), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.text(6, 1.4, "Saídas: stems.csv | plot_summary.csv | review_list.csv | stems_labeled.laz (tree_id) | run.json | este relatório", ha="center", fontsize=7.5)
    save(fig, "g13_fluxo_pipeline.png")

    # ---------------- comparacao com a referencia
    if ref is not None and ref.get("pareamento") is not None and len(ref["pareamento"]):
        ann, P = ref["ann"], ref["pareamento"].copy()
        # c1 planta: pipeline x anotacoes
        fig, ax = plt.subplots(figsize=(8.4, 7.8))
        _nuvem_fundo(ax, c, alpha=.3, bins=0.12)
        usados = set(P.stem_idx.astype(int))
        for i, r in s.iterrows():
            ax.add_patch(Circle((r.x_utm, r.y_utm), max(r.dbh_cm / 200, 0.04), fc="none", ec=TEAL if i in usados else ORG, lw=1.3, zorder=3))
        pareados_ann = set(P.rotulo)
        for r in ann.itertuples():
            det = r.lab in pareados_ann
            ax.scatter(r.x, r.y, marker="+" if det else "x", s=45, c=NAVY if det else RED, zorder=4)
        ax.scatter([], [], marker="o", facecolor="none", edgecolor=TEAL, label="fuste pareado com anotação")
        ax.scatter([], [], marker="o", facecolor="none", edgecolor=ORG, label="fuste sem anotação")
        ax.scatter([], [], marker="+", c=NAVY, label="anotação detectada")
        ax.scatter([], [], marker="x", c=RED, label="anotação NÃO detectada")
        ax.legend(loc="upper left", fontsize=7)
        _base(ax, c, "Pipeline x anotações do ReCap")
        save(fig, "c1_planta_comparacao.png")
        V = P[(P.cond_ann == "v") & (P.cond_man == "v") & P.dap_man.notna()].copy()
        if len(V) > 2:
            V["err"] = V.dbh_pipe - V.dap_man
            fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
            for ax, (a, b, t) in zip(axs[:2], ((V.dap_man, V.dbh_pipe, "DAP (cm)"), (V.h_man, V.h_pipe, "Altura (m)"))):
                ok = a.notna() & b.notna()
                if ok.sum() == 0:
                    ax.set_title(t + " (sem dados)")
                    continue
                ax.scatter(a[ok], b[ok], s=24, c=TEAL, edgecolor="k", lw=.3)
                lim = [0, max(a[ok].max(), b[ok].max()) * 1.05]
                ax.plot(lim, lim, "--", c=GRAY)
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                e = (b[ok] - a[ok]).values
                ax.text(.04, .94, f"n={ok.sum()}\nviés {e.mean():+.2f}\nRMSE {np.sqrt((e**2).mean()):.2f}", transform=ax.transAxes, va="top", fontsize=8)
                ax.set_xlabel("TLS manual (ReCap)")
                ax.set_ylabel("Pipeline")
                ax.set_title(t)
            axs[2].scatter(V.dap_man, V.err, s=24, c=NAVY)
            axs[2].axhline(0, c=GRAY, ls="--")
            axs[2].set_xlabel("DAP manual (cm)")
            axs[2].set_ylabel("erro do DAP (pipeline menos manual, cm)")
            axs[2].set_title("Erro de DAP por tamanho")
            save(fig, "c2_1a1.png")
            # erro por classe
            bins = [0, 5, 10, 20, 200]
            lab = ["< 5", "5 a 10", "10 a 20", ">= 20"]
            V["cls"] = pd.cut(V.dap_man, bins, labels=lab, right=False)
            g = V.groupby("cls", observed=False).err.agg(["mean", "count"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(g)), g["mean"].fillna(0), color=TEAL)
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([f"{i}\n(n={int(n)})" for i, n in zip(g.index, g["count"])])
            ax.axhline(0, c=GRAY)
            ax.set_xlabel("DAP manual (cm)")
            ax.set_ylabel("viés médio do DAP (cm)")
            ax.set_title("Viés do DAP por classe de tamanho", fontsize=9)
            save(fig, "c3_vies_por_classe.png")
            # mapa de erro
            fig, ax = plt.subplots(figsize=(8.4, 7.8))
            _nuvem_fundo(ax, c, alpha=.3, bins=0.12)
            lim = max(4.0, float(np.nanpercentile(np.abs(V.err), 90)))
            m = ax.scatter(V.x_pipe, V.y_pipe, c=V.err, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), s=90, edgecolor="k", lw=.5, zorder=3) if "x_pipe" in V else None
            if m is not None:
                plt.colorbar(m, ax=ax, shrink=.75, label="DAP pipeline menos manual (cm)")
                for r in V.itertuples():
                    ax.text(r.x_pipe + .1, r.y_pipe + .1, r.rotulo.replace(" viva", ""), fontsize=5.5)
                _base(ax, c, "Erro de DAP por fuste pareado (vermelho = pipeline maior)")
                save(fig, "c4_mapa_erro_dap.png")
            else:
                plt.close(fig)
            # cortes dos maiores erros
            top = V.reindex(V.err.abs().sort_values(ascending=False).index).head(8)
            P13 = c.z["corte_1"].astype(float) + c.o[:2]
            fig, axs = plt.subplots(2, 4, figsize=(12, 6.4))
            for ax, r in zip(axs.ravel(), top.itertuples()):
                if "x_pipe" not in V:
                    break
                mm = (np.abs(P13[:, 0] - r.x_pipe) < 0.7) & (np.abs(P13[:, 1] - r.y_pipe) < 0.7)
                ax.scatter(P13[mm, 0], P13[mm, 1], s=3, c=GRAY)
                ax.add_patch(Circle((r.x_pipe, r.y_pipe), r.dbh_pipe / 200, fc="none", ec=TEAL, lw=1.4))
                ax.add_patch(Circle((r.x_pipe, r.y_pipe), r.dap_man / 200, fc="none", ec=NAVY, lw=1.4, ls="--"))
                ax.set_xlim(r.x_pipe - .7, r.x_pipe + .7)
                ax.set_ylim(r.y_pipe - .7, r.y_pipe + .7)
                ax.set_aspect("equal")
                ax.set_title(f"{r.rotulo}\npipeline {r.dbh_pipe:.1f} (verde) x manual {r.dap_man:.1f} cm (tracejado)", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle("Corte a 1,3 m nos 8 fustes de maior erro de DAP. Os círculos ficam na posição do fuste na altura de medição", fontsize=9)
            save(fig, "c5_cortes_maiores_erros.png")
    return feitas


# ------------------------------------------------------------------ PDF
def _tab(rows, widths, cell, cellb):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    data = [[Paragraph(str(v), cellb if i == 0 else cell) for v in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFC8D0")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY))]
    st += [("BACKGROUND", (0, i), (0, i), colors.HexColor("#E8EEF3")) for i in range(1, len(rows))]
    t.setStyle(TableStyle(st))
    return t


def pdf(out, nome, stems, params=None, area=None, area_origem="", plot_summary=None, avaliacao=None, ref_texto=None, comando=None, versao=""):
    """Monta o PDF a partir das figuras ja geradas em out/figuras."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, KeepTogether, PageBreak
    from PIL import Image as PI
    out = pathlib.Path(out)
    fd = out / "figuras"
    navy, teal, gray = colors.HexColor(NAVY), colors.HexColor(TEAL), colors.HexColor("#5C6670")
    ss = {
        "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=navy, spaceAfter=4),
        "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=10, leading=13, textColor=gray, spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=navy, spaceBefore=8, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=teal, spaceBefore=6, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="Helvetica", fontSize=9.3, leading=12.8, alignment=TA_JUSTIFY, spaceAfter=4),
        "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10.5, textColor=gray, spaceAfter=9, alignment=TA_LEFT),
        "warn": ParagraphStyle("w", fontName="Helvetica", fontSize=9, leading=12.3, backColor=colors.HexColor("#FFF4E0"), borderColor=colors.HexColor("#E08E0B"),
                               borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=9),
        "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.2, leading=10.3),
        "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.2, leading=10.3, textColor=colors.white),
    }
    P = lambda t, k="p": Paragraph(t, ss[k])
    t = lambda rows, w: _tab(rows, w, ss["cell"], ss["cellb"])

    def fig(name, cap, maxh=19.0):
        path = fd / name
        if not path.exists():
            return []
        w0, h0 = PI.open(path).size
        w = 17 * cm
        h = w * h0 / w0
        if h > maxh * cm:
            h = maxh * cm
            w = h * w0 / h0
        return [KeepTogether([Image(str(path), width=w, height=h), Spacer(1, 3), Paragraph(cap, ss["cap"])])]

    def footer(cv, d):
        cv.saveState()
        cv.setFont("Helvetica", 7.5)
        cv.setFillColor(gray)
        cv.drawString(2 * cm, 1.1 * cm, f"tls_mangue {versao} | parcela {nome}")
        cv.drawRightString(19 * cm, 1.1 * cm, f"página {d.page}")
        cv.restoreState()

    dest = out / f"relatorio_{nome}.pdf"
    doc = BaseDocTemplate(str(dest), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=f"Relatório {nome}")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(2 * cm, 1.8 * cm, 17 * cm, A4[1] - 3.6 * cm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
    s = stems
    n_rev = int(s["flags"].fillna("").str.contains("REVIEW").sum())
    S = [P(f"Parcela {nome}: relatório do pipeline", "title"), P("Extração de DAP, altura, densidade e área basal a partir de nuvem TLS (Pipeline C, tls_mangue)", "sub")]
    S += [P("<b>Leia antes de usar os números.</b> O pipeline ainda não separa árvore viva de morta nem distingue espécie, então densidade e área basal incluem todos os fustes detectados. "
            "Quando há comparação, a referência é o TLS manual feito no ReCap (planilha de extração), que não é o campo: o erro mostrado é a distância entre o pipeline e o fluxo manual, "
            "e não o erro verdadeiro do DAP.", "warn")]
    S += [P("1. Resumo", "h1")]
    A = area
    rows = [["Item", "Valor"], ["Fustes detectados" + (" (dentro do polígono)" if "polígono" in area_origem else ""), len(s)],
            ["Fustes marcados REVIEW", f"{n_rev} ({100 * n_rev / max(len(s), 1):.0f}%)"],
            ["Área usada nos totais por hectare", f"{A:.1f} m<super>2</super> ({area_origem})" if A else "não informada"],
            ["DAP médio (cm)", f"{s.dbh_cm.mean():.1f}"], ["Altura média (m)", f"{s.height_m.mean():.1f}" if s.height_m.notna().any() else "-"]]
    if A:
        ha = A / 1e4
        ok = ~s["flags"].fillna("").str.contains("REVIEW")
        rows += [["Densidade (fustes/ha), todos | sem REVIEW", f"{len(s) / ha:,.0f} | {ok.sum() / ha:,.0f}"],
                 ["Área basal (m<super>2</super>/ha), todos | sem REVIEW", f"{s.basal_area_m2.sum() / ha:.1f} | {s.basal_area_m2[ok].sum() / ha:.1f}"]]
    S += [t(rows, [8, 9])]
    if avaliacao:
        d = avaliacao
        S += [P("Comparação com o TLS manual (ReCap)", "h2")]
        rr = [["Métrica", "Valor"], ["Anotações vivas no polígono | detectadas", f"{d['anot_vivas_no_poligono']} | {d['anot_vivas_detectadas']} ({100 * d['recall_vivas']:.0f}%)"],
              ["Fustes do pipeline no polígono | indivíduos da planilha (vivos | mortos)", f"{d['fustes_pipeline_no_poligono']} | {d['individuos_planilha']} ({d['planilha_vivos']} | {d['planilha_mortos']})"]]
        for k, lab, u in (("dap", "DAP", "cm"), ("altura", "Altura", "m")):
            if k in d and d[k]:
                m = d[k]
                rr.append([f"{lab} (n={m['n']}): viés | RMSE | mediana do erro absoluto | r", f"{m['vies']:+.2f} | {m['rmse']:.2f} | {m.get('mediana_erro_abs', float('nan')):.2f} {u} | {m.get('r', float('nan')):.2f}"])
        tt = d["totais"]
        mv, pp = tt["manual_vivos"], tt["pipeline_todos_no_poligono"]
        rr += [["Área basal (m<super>2</super>/ha): manual vivos | pipeline todos | pipeline sem REVIEW",
                f"{mv['area_basal_m2ha']:.1f} | {pp['area_basal_m2ha']:.1f} | {tt['pipeline_sem_REVIEW']['area_basal_m2ha']:.1f}" if "pipeline_sem_REVIEW" in tt else f"{mv['area_basal_m2ha']:.1f} | {pp['area_basal_m2ha']:.1f}"],
               ["Densidade (fustes/ha): manual vivos | pipeline todos", f"{mv['densidade_ha']:,.0f} | {pp['densidade_ha']:,.0f}"]]
        S += [t(rr, [10, 7])]
        if ref_texto:
            S += [P(ref_texto, "cap")]
    S += [P("Flags presentes", "h2")]
    fl = s["flags"].fillna("").str.split(";").explode()
    fl = fl[fl != ""].value_counts()
    S += [t([["Flag", "Fustes", "Significado"]] + [[k, int(v), FLAGS_DOC.get(k, "")] for k, v in fl.items()], [4.2, 1.4, 11.4])] if len(fl) else [P("Nenhum fuste com flags.")]
    S += [PageBreak(), P("2. Parâmetros desta execução", "h1")]
    if params:
        from .config import Params
        dflt = Params().__dict__
        mud = [[k, params[k], dflt.get(k, "")] for k in params if k in dflt and params[k] != dflt[k]]
        S += [P("Parâmetros diferentes do padrão do pacote (os demais estão no padrão; a lista completa está em run.json):")]
        S += [t([["Parâmetro", "Usado", "Padrão"]] + mud, [6, 5, 6])] if mud else [P("Todos os parâmetros no padrão.")]
    S += [P("Parâmetros que mais afetam o resultado em manguezal: voxel_size, slice_min/slice_max, r_min/r_max, ransac_tolerance, min_arc_deg, min_chain_slices, root_ratio, "
            "dbh_window e track_radius. Calibre em mais de uma parcela e valide em outra.")]
    S += [P("3. A nuvem sem marcações", "h1")]
    S += fig("g13_fluxo_pipeline.png", "Figura 1. Fluxo do pipeline.", 6)
    S += fig("g01_planta_nuvem.png", "Figura 2. Nuvem em planta.", 15)
    S += fig("g02_vistas_laterais.png", "Figura 3. Vistas laterais, cor = altura acima do solo.", 15)
    S += fig("g03_perfil_vertical.png", "Figura 4. Perfil vertical de pontos.", 8)
    S += fig("g04_mdt.png", "Figura 5. MDT e densidade de pontos. Áreas de MDT que sobem de repente costumam ser vegetação baixa tomada como solo; lacunas na densidade são água ou oclusão.", 8)
    S += fig("g05_altura_dossel.png", "Figura 6. Altura máxima por célula.", 14)
    S += [PageBreak(), P("4. Árvores detectadas", "h1")]
    S += fig("g06_cortes_horizontais.png", "Figura 7. Cortes horizontais em quatro alturas, com os fustes detectados.", 17)
    S += fig("g07_arvores_dap.png", "Figura 8. DAP por fuste (número = stem_id da tabela stems.csv).", 16)
    S += fig("g08_arvores_altura.png", "Figura 9. Altura por fuste.", 16)
    S += fig("g09_arvores_qualidade.png", "Figura 10. Qualidade do ajuste e fustes REVIEW.", 16)
    S += fig("g10_distribuicoes.png", "Figura 11. Distribuições de DAP e altura.", 7)
    S += fig("g11_flags_qualidade.png", "Figura 12. Flags e qualidade.", 7)
    S += fig("g12_exemplos_fustes.png", "Figura 13. Exemplos de fustes em vista lateral.", 11)
    if (fd / "c1_planta_comparacao.png").exists():
        S += [PageBreak(), P("5. Comparação com o TLS manual (ReCap)", "h1")]
        S += fig("c1_planta_comparacao.png", "Figura 14. Fustes do pipeline e anotações. Laranja = fuste sem anotação (candidato a falso positivo ou árvore não anotada); x vermelho = anotação não detectada.", 17)
        S += fig("c2_1a1.png", "Figura 15. Pipeline x TLS manual, somente vivos pareados.", 8)
        S += fig("c3_vies_por_classe.png", "Figura 16. Viés do DAP por classe de tamanho.", 7)
        S += fig("c4_mapa_erro_dap.png", "Figura 17. Mapa do erro de DAP.", 15)
        S += fig("c5_cortes_maiores_erros.png", "Figura 18. Cortes a 1,3 m nos fustes de maior erro. Verde = pipeline, tracejado = manual.", 12)
    S += [PageBreak(), P("6. Fustes para revisar", "h1")]
    rv = s[s["flags"].fillna("").str.contains("REVIEW")].sort_values("quality").head(25)
    if len(rv):
        S += [P("Os 25 de menor qualidade. A lista completa está em review_list.csv; os tree_id estão em stems_labeled.laz (abrir no CloudCompare ou Segfix).")]
        S += [t([["stem_id", "x", "y", "DAP (cm)", "altura (m)", "qualidade", "flags"]] + [[int(r.stem_id), f"{r.x_utm:.2f}", f"{r.y_utm:.2f}", f"{r.dbh_cm:.1f}",
              "-" if not np.isfinite(r.height_m) else f"{r.height_m:.1f}", f"{r.quality:.2f}", str(r.flags)] for r in rv.itertuples()], [1.4, 1.7, 1.7, 1.6, 1.8, 1.6, 7.2])]
    else:
        S += [P("Nenhum fuste marcado REVIEW.")]
    S += [P("7. Limitações e como reproduzir", "h1")]
    S += [P("Sem separação de vivo e morto nem de espécie; troncos do mesmo indivíduo contam como fustes separados; a altura é a do topo do fuste rastreado, que pode parar antes do topo da árvore "
            "ou entrar na copa de uma vizinha; fustes abaixo de 2 * r_min de diâmetro não são detectados; resultados dependem da exportação e do registro da nuvem.")]
    if comando:
        S += [P(f"Comando: <font face='Courier' size='8'>{comando}</font>")]
    S += [P("Arquivos desta execução: stems.csv, plot_summary.csv, review_list.csv, stems_labeled.laz, run.json, relatorio_dados.npz e a pasta figuras.", "cap")]
    doc.build(S)
    return dest


def gerar(out, nome, stems, poly=None, params=None, area=None, area_origem="", versao="", comando=None, ref=None, avaliacao=None, ref_texto=None):
    """Figuras + PDF. Retorna o caminho do PDF."""
    figuras(out, stems, poly=poly, nome=nome, ref=ref)
    return pdf(out, nome, stems, params=params, area=area, area_origem=area_origem, avaliacao=avaliacao, ref_texto=ref_texto, comando=comando, versao=versao)
