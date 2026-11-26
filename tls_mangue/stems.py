"""Etapas 3 e 4: deteccao de fustes em varias fatias, filtro de raizes-escora, DAP adaptativo e altura.

Logica pensada para Rhizophora mangle:
  * circulos sao ajustados por RANSAC em varias fatias de altura (nao so a 1,30 m);
  * circulos de fatias vizinhas sao ligados em cadeias (fuste = cadeia continua, permite inclinacao);
  * cadeias finas que terminam encostadas em um tronco mais grosso sao tratadas como raizes-escora e descartadas;
  * a altura de medicao do DAP sobe para acima da raiz mais alta quando ela passa de 1,30 m (modo adaptativo);
  * cada fuste recebe um indice de qualidade e flags para a revisao humana.
"""
import math
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------- circulos
def _circle_from_3(p3):
    (x1, y1), (x2, y2), (x3, y3) = p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


def _refine(xy, cx, cy, r):
    def res(q):
        return np.hypot(xy[:, 0] - q[0], xy[:, 1] - q[1]) - q[2]
    sol = least_squares(res, [cx, cy, r], method="lm")
    return sol.x[0], sol.x[1], abs(sol.x[2])


def ransac_circle(xy, p, rng):
    """Circulo por RANSAC com limites de raio. Retorna dict ou None."""
    n = len(xy)
    if n < p.min_inliers:
        return None
    if n > 800:
        sub = rng.choice(n, 800, replace=False)
    else:
        sub = np.arange(n)
    best = None
    for _ in range(p.ransac_iterations):
        idx = rng.choice(sub, 3, replace=False)
        c = _circle_from_3(xy[idx])
        if c is None or not (p.r_min <= c[2] <= p.r_max):
            continue
        d = np.abs(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - c[2])
        k = int((d < p.ransac_tolerance).sum())
        if best is None or k > best[0]:
            best = (k, c)
    if best is None or best[0] < p.min_inliers:
        return None
    cx, cy, r = best[1]
    d = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
    inl = d < p.ransac_tolerance
    cx, cy, r = _refine(xy[inl], cx, cy, r)
    if not (p.r_min <= r <= p.r_max):
        return None
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    inl = np.abs(d - r) < p.ransac_tolerance
    if inl.sum() < p.min_inliers:
        return None
    rmse = float(np.sqrt(np.mean((d[inl] - r) ** 2)))
    ang = np.degrees(np.arctan2(xy[inl, 1] - cy, xy[inl, 0] - cx)) % 360
    arc = 10.0 * len(np.unique((ang // 10).astype(int)))
    inner = float((d < 0.6 * r).sum()) / max(1, int((d < 1.3 * r).sum()))
    if rmse > p.max_circle_rmse or arc < p.min_arc_deg or inner > p.max_inner_fraction:
        return None
    return {"x": cx, "y": cy, "r": r, "rmse": rmse, "arc": arc, "n": int(inl.sum()), "inl": inl}


def circles_in_slice(xy, p, rng):
    """Todos os circulos validos de uma fatia (ate max_circles_per_cluster por aglomerado)."""
    if len(xy) < p.cluster_min_samples:
        return []
    lab = DBSCAN(eps=p.cluster_eps, min_samples=p.cluster_min_samples).fit_predict(xy)
    out = []
    for c in np.unique(lab[lab >= 0]):
        pts = xy[lab == c]
        for _ in range(p.max_circles_per_cluster):
            if len(pts) < p.min_inliers:
                break
            circ = ransac_circle(pts, p, rng)
            if circ is None:
                break
            out.append({k: v for k, v in circ.items() if k != "inl"})
            pts = pts[~circ["inl"]]
    return out


# ---------------------------------------------------------------- cadeias
def build_chains(xyz_n, p, seed=0):
    """xyz_n: coordenadas com z = altura acima do solo. Retorna lista de cadeias (dicts de listas)."""
    rng = np.random.default_rng(seed)
    heights = np.arange(p.slice_min, p.slice_max + 1e-9, p.slice_step)
    chains = []
    for h in heights:
        m = np.abs(xyz_n[:, 2] - h) <= p.slice_thickness / 2
        cands = circles_in_slice(xyz_n[m, :2], p, rng) if m.sum() >= p.cluster_min_samples else []
        active = [c for c in chains if c["miss"] <= p.max_gap_slices]
        if active and cands:
            cost = np.full((len(active), len(cands)), 1e6)
            for i, ch in enumerate(active):
                px, py = _predict(ch, h)
                allow = p.link_dist + 0.05 * ch["miss"]
                for j, cd in enumerate(cands):
                    dist = math.hypot(cd["x"] - px, cd["y"] - py)
                    rl = ch["r"][-1]
                    if dist <= allow and abs(cd["r"] - rl) <= max(0.04, 0.6 * rl):
                        cost[i, j] = dist
            ri, ci = linear_sum_assignment(cost)
            used_c = set()
            matched = set()
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e5:
                    _append(active[i], h, cands[j])
                    used_c.add(j)
                    matched.add(i)
            for i, ch in enumerate(active):
                if i not in matched:
                    ch["miss"] += 1
            for j, cd in enumerate(cands):
                if j not in used_c:
                    chains.append(_new_chain(h, cd))
        else:
            for ch in active:
                ch["miss"] += 1
            for cd in cands:
                chains.append(_new_chain(h, cd))
    return chains


def _new_chain(h, cd):
    ch = {"z": [], "x": [], "y": [], "r": [], "rmse": [], "arc": [], "miss": 0}
    _append(ch, h, cd)
    return ch


def _append(ch, h, cd):
    ch["z"].append(h); ch["x"].append(cd["x"]); ch["y"].append(cd["y"]); ch["r"].append(cd["r"])
    ch["rmse"].append(cd["rmse"]); ch["arc"].append(cd["arc"]); ch["miss"] = 0


def _predict(ch, h):
    z, x, y = np.array(ch["z"][-3:]), np.array(ch["x"][-3:]), np.array(ch["y"][-3:])
    if len(z) >= 2 and z[-1] != z[0]:
        bx = np.polyfit(z, x, 1); by = np.polyfit(z, y, 1)
        return np.polyval(bx, h), np.polyval(by, h)
    return x[-1], y[-1]


def _axis(ch):
    z = np.array(ch["z"]); x = np.array(ch["x"]); y = np.array(ch["y"])
    bx = np.polyfit(z, x, 1); by = np.polyfit(z, y, 1)
    return bx, by


def filter_chains(chains, p):
    """Remove cadeias curtas, muito inclinadas e raizes-escora (cadeias finas que encostam em um tronco mais grosso)."""
    keep = []
    for ch in chains:
        n = len(ch["z"])
        if n < p.min_chain_slices:
            continue
        bx, by = _axis(ch)
        lean = math.degrees(math.atan(math.hypot(bx[0], by[0])))
        if lean > p.max_lean_deg:
            continue
        ch["lean_deg"] = lean
        ch["r_med"] = float(np.median(ch["r"]))
        keep.append(ch)
    out = []
    for a in keep:
        if not _is_root(a, keep, p):
            out.append(a)
    # duplicatas: cadeias que coincidem em alguma fatia comum sao o mesmo fuste (ex.: raiz que se funde ao tronco)
    out.sort(key=lambda c: -len(c["z"]))
    final = []
    for c in out:
        if not any(_coincide(c, f, p) for f in final):
            final.append(c)
    return final


def _is_root(a, others, p):
    """Cadeia fina cujo eixo, prolongado para cima, converge para um tronco mais grosso = raiz-escora."""
    za = a["z"][-1]
    bxa, bya = _axis(a)
    xa, ya = np.polyval(bxa, za), np.polyval(bya, za)
    zs = np.linspace(za, za + p.root_extrapolate, 7)
    for b in others:
        if b is a or b["r_med"] < p.root_radius_ratio * a["r_med"] or b["z"][-1] < za + p.slice_step:
            continue
        bxb, byb = _axis(b)
        d = np.hypot(np.polyval(bxa, zs) - np.polyval(bxb, zs), np.polyval(bya, zs) - np.polyval(byb, zs))
        lim = b["r_med"] + a["r_med"] + p.root_attach_margin
        if d.min() <= lim and d[0] > d[-1] - 1e-9 or d[0] <= lim:
            return True
    return False


def _coincide(a, b, p):
    """True se, em alguma altura comum, os circulos de a e b se sobrepoem (distancia < 0,6 x o maior raio)
    ou se os eixos estao a menos de merge_radius."""
    zb = {round(z, 6): i for i, z in enumerate(b["z"])}
    for i, z in enumerate(a["z"]):
        j = zb.get(round(z, 6))
        if j is not None:
            d = math.hypot(a["x"][i] - b["x"][j], a["y"][i] - b["y"][j])
            if d < max(0.6 * max(a["r"][i], b["r"][j]), 0.0) or d < p.merge_radius * 0.5:
                return True
    zc = float(np.median(a["z"]))
    xa, ya = np.polyval(_axis(a)[0], zc), np.polyval(_axis(a)[1], zc)
    xb, yb = np.polyval(_axis(b)[0], zc), np.polyval(_axis(b)[1], zc)
    return math.hypot(xa - xb, ya - yb) < p.merge_radius


# ---------------------------------------------------------------- DAP e altura
def _root_zone_top(z, r, p):
    """Topo da zona de raiz, ou nan. So ha evidencia de raiz quando o raio NA JANELA do DAP padrao
    (dbh_height +- dbh_halfwindow) e maior que root_ratio x o raio estavel do topo da cadeia. Nesse caso o bloco
    de fatias alargadas que contem a janela (tolerando uma fatia falha) define o topo da zona de raiz.
    Sem fatia na janela, ou com raio normal nela, nao sobe: raio alto em outra altura (vizinho colado, ramo)
    nao e raiz e nao deve mudar a altura de medicao."""
    nref = min(p.stable_window, len(r))
    r_ref = float(np.median(r[-nref:]))
    over = r > p.root_ratio * r_ref
    win = np.where(np.abs(z - p.dbh_height) <= p.dbh_halfwindow + 1e-9)[0]
    if len(win) == 0 or not over[win].any():
        return np.nan
    if float(np.median(r[win])) <= p.root_ratio * r_ref and over[win].sum() < 1 + len(win) // 2:
        return np.nan
    k = int(win[over[win]].max()); last, miss = k, 0
    j = k + 1
    while j < len(z):
        if over[j]:
            last, miss = j, 0
        else:
            miss += 1
            if miss > 1:
                break
        j += 1
    return float(z[last])


def _dbh_from_window(z, r, arc, rm, lo, hi, p):
    """Mediana robusta do raio das fatias em [lo, hi], so com fatias de arco suficiente quando houver."""
    sel = (z >= lo - 1e-9) & (z <= hi + 1e-9)
    flags = []
    if sel.sum() == 0:
        return None, flags
    good = sel & (arc >= p.min_arc_dbh)
    if good.any():
        use = good
    else:
        use = sel
        flags.append("LOW_ARC")
    # rejeita fatias cujo raio foge da mediana (vizinho colado, ramo)
    if use.sum() >= 3:
        med = np.median(r[use])
        ok = use & (np.abs(r - med) <= p.dbh_outlier_frac * med)
        if ok.sum() >= 2:
            use = ok
    return use, flags


def measure_stem(ch, xyz_n, p):
    """DAP (cm), altura de medicao, altura, qualidade e flags de um fuste."""
    z = np.array(ch["z"]); r = np.array(ch["r"]); rm = np.array(ch["rmse"]); arc = np.array(ch["arc"])
    flags = []
    h_meas = p.dbh_height
    z_root = np.nan
    if p.adaptive_dbh:
        z_root = _root_zone_top(z, r, p)
        if np.isfinite(z_root):
            h_meas = max(p.dbh_height, z_root + p.root_margin)
    adjusted = h_meas > p.dbh_height + p.root_flag_tol
    if adjusted:
        flags.append("ROOT_ADJUSTED")
        use, f2 = _dbh_from_window(z, r, arc, rm, h_meas, h_meas + p.dbh_window, p)
    else:
        use, f2 = _dbh_from_window(z, r, arc, rm, h_meas - p.dbh_halfwindow, h_meas + p.dbh_halfwindow, p)
    if h_meas > p.max_dbh_height:
        flags.append("HIGH_DBH_HEIGHT")
    if use is None:
        # sem fatia perto da altura de medicao: usa a mais proxima, mas so sem ser silencioso
        k = int(np.argmin(np.abs(z - h_meas)))
        use = np.zeros(len(z), bool); use[k] = True
        f2 = []
        flags.append("NO_SLICE_AT_DBH")
        if abs(z[k] - h_meas) > p.max_dbh_offset:
            flags.append("DBH_EXTRAPOLATED")
    flags += f2
    sel = use
    r_sel = r[sel]
    dbh_cm = 200.0 * float(np.median(r_sel))
    cv = float(np.std(r_sel) / np.mean(r_sel)) if len(r_sel) > 1 else np.nan
    bx, by = _axis(ch)
    zm = float(np.clip(h_meas, z.min(), z.max()))
    x, y = float(np.polyval(bx, zm)), float(np.polyval(by, zm))
    height, height_occ = _stem_height(xyz_n, bx, by, zm, z.max(), p)
    q_n = min(1.0, len(z) / 8.0)
    q_fit = max(0.0, 1.0 - float(np.mean(rm[sel])) / p.max_circle_rmse)
    q_arc = min(1.0, float(np.mean(arc[sel])) / 360.0 / 0.6)
    q_cv = 1.0 if (not np.isfinite(cv)) else max(0.0, 1.0 - cv / p.stable_cv / 2.0)
    q = 0.25 * (q_n + q_fit + q_arc + q_cv)
    if "NO_SLICE_AT_DBH" in flags:
        q = min(q, p.min_quality - 0.01)
    if len(z) < 6:
        flags.append("FEW_SLICES")
    if float(np.mean(arc[sel])) < 180 and "LOW_ARC" not in flags:
        flags.append("LOW_ARC")
    if np.isfinite(cv) and cv > p.stable_cv:
        flags.append("UNSTABLE_RADIUS")
    fl = set(flags)
    if (q < p.min_quality or "DBH_EXTRAPOLATED" in fl or "HIGH_DBH_HEIGHT" in fl
            or ("FEW_SLICES" in fl and "LOW_ARC" in fl)):
        flags.append("REVIEW")
    return {
        "x": x, "y": y, "dbh_cm": dbh_cm, "dbh_height_m": float(h_meas), "root_zone_top_m": z_root,
        "height_m": height, "height_occ_m": height_occ, "basal_area_m2": math.pi * (dbh_cm / 200.0) ** 2,
        "quality": float(q), "n_slices": int(len(z)), "n_slices_dbh": int(sel.sum()),
        "z_min": float(z.min()), "z_max": float(z.max()),
        "lean_deg": float(ch["lean_deg"]), "circle_rmse_cm": 100.0 * float(np.mean(rm[sel])),
        "arc_deg": float(np.mean(arc[sel])), "radius_cv": cv, "flags": ";".join(dict.fromkeys(flags)),
        "_bx": bx, "_by": by, "_r": float(np.median(r_sel)),
    }


def _stem_height(xyz_n, bx, by, z0, z_chain_top, p):
    """Altura da arvore. Retorna (altura_por_rastreamento, altura_por_ocupacao_antiga).
    Rastreamento: sobe em degraus de track_step, recentrando o eixo na mediana dos pontos do degrau
    (segue fustes tortos e inclinados em vez de extrapolar a reta ajustada em 0,5 a 4 m). Para quando ha lacuna
    maior que height_gap. A altura e o percentil height_pct dos pontos do ultimo trecho ocupado."""
    cx = float(np.polyval(bx, z0)); cy = float(np.polyval(by, z0))
    box = (np.abs(xyz_n[:, 0] - cx) < p.track_box) & (np.abs(xyz_n[:, 1] - cy) < p.track_box) & (xyz_n[:, 2] >= z0 - 0.1)
    pts = xyz_n[box]
    if len(pts) == 0:
        return np.nan, np.nan
    # --- altura antiga (ocupacao em torno do eixo ajustado), mantida para comparacao
    dx = pts[:, 0] - np.polyval(bx, pts[:, 2]); dy = pts[:, 1] - np.polyval(by, pts[:, 2])
    zz = pts[np.hypot(dx, dy) <= p.height_radius, 2]
    h_occ = np.nan
    if len(zz):
        nb = int(np.ceil((zz.max() - z0) / p.height_bin)) + 1
        hist, _ = np.histogram(zz, bins=nb, range=(z0, z0 + nb * p.height_bin))
        occ = hist >= p.height_min_points
        gb = int(round(p.height_gap / p.height_bin)); top = -1; gap = 0
        for i, o in enumerate(occ):
            if o:
                top, gap = i, 0
            else:
                gap += 1
                if gap > gb:
                    break
        h_occ = float(z0 + (top + 1) * p.height_bin) if top >= 0 else np.nan
    # --- rastreamento
    step = p.track_step
    zl = z0; c = np.array([cx, cy]); slope = np.array([bx[0], by[0]])
    smax = math.tan(math.radians(p.max_lean_deg))
    n = np.hypot(*slope)
    if n > smax:
        slope = slope * smax / n
    hist_c = [(z0, c.copy())]
    top_pts = None; gap = 0.0; top_z = np.nan
    zmax = pts[:, 2].max()
    while zl < zmax + step:
        pred = c + slope * step
        m = (pts[:, 2] >= zl) & (pts[:, 2] < zl + step)
        pp = pts[m]
        d = np.hypot(pp[:, 0] - pred[0], pp[:, 1] - pred[1])
        pp = pp[d <= p.track_radius]
        zl += step
        if len(pp) >= p.track_min_points:
            cen = np.median(pp[:, :2], axis=0)
            c = 0.5 * pred + 0.5 * cen
            hist_c.append((zl, c.copy()))
            if len(hist_c) >= 4:
                a, b2 = hist_c[-4], hist_c[-1]
                sl = (b2[1] - a[1]) / (b2[0] - a[0])
                nn = np.hypot(*sl)
                slope = sl if nn <= smax else sl * smax / nn
            top_pts = pp; gap = 0.0; top_z = np.percentile(pp[:, 2], p.height_pct)
            top_z = max(top_z, zl - step)
        else:
            c = pred; gap += step
            if gap > p.height_gap:
                break
    return (float(top_z) if np.isfinite(top_z) else np.nan), h_occ


def detect_stems(xyz_n, p, seed=0):
    """Pipeline completo de fustes. xyz_n: (N,3) com z = altura acima do solo."""
    chains = build_chains(xyz_n, p, seed)
    n_raw = len(chains)
    chains = filter_chains(chains, p)
    stems = [measure_stem(c, xyz_n, p) for c in chains]
    stems.sort(key=lambda s: (s["x"], s["y"]))
    for i, s in enumerate(stems, 1):
        s["stem_id"] = i
    return stems, {"chains_raw": n_raw, "stems": len(stems)}


def label_points(xyz_n, stems, p):
    """tree_id por ponto (-1 = sem fuste), para revisao no CloudCompare ou Segfix. Considera 0,3 m a slice_max + 2 m."""
    lab = np.full(len(xyz_n), -1, dtype=np.int32)
    if not stems:
        return lab
    zmask = (xyz_n[:, 2] >= 0.3) & (xyz_n[:, 2] <= p.slice_max + 2.0)
    idx = np.where(zmask)[0]
    best = np.full(len(idx), np.inf)
    for s in stems:
        ax = np.polyval(s["_bx"], xyz_n[idx, 2]); ay = np.polyval(s["_by"], xyz_n[idx, 2])
        d = np.hypot(xyz_n[idx, 0] - ax, xyz_n[idx, 1] - ay)
        lim = max(0.15, p.label_radius_factor * s["_r"])
        upd = (d <= lim) & (d < best)
        best[upd] = d[upd]
        lab[idx[upd]] = s["stem_id"]
    return lab
