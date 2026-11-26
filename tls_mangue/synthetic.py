"""Nuvem sintetica de manguezal para TESTAR O CODIGO (nao e dado cientifico).

Simula parcela de ~15 x 16 m com Rhizophora (raizes-escora que se inserem entre 0,8 e 2,4 m),
Avicennia e Laguncularia, solo com canal sem retorno e pontos de reflexo abaixo do solo,
folhagem e quatro posicoes de scanner (cobertura so da face visivel + erro de registro de 4 mm).
Retorna a tabela de verdade (DAP medido logo acima da raiz mais alta, como em Jackson et al. 2026).
"""
import numpy as np
import pandas as pd


def _cyl(rng, p0, p1, r, dens):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    ax = (p1 - p0) / L
    n = max(8, int(2 * np.pi * r * L * dens))
    t = rng.random(n) * L
    th = rng.random(n) * 2 * np.pi
    a = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    nrm = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b
    pts = p0 + t[:, None] * ax + r * nrm
    return pts, nrm


def make_plot(seed=1, n_stems=40, size=(15.0, 16.0), dens=2500, out_las=None):
    rng = np.random.default_rng(seed)
    W, H = size
    scanners = np.array([[1.5, 1.5, 1.5], [W - 1.5, 1.5, 1.5], [1.5, H - 1.5, 1.5], [W - 1.5, H - 1.5, 1.5]])
    allp, truth = [], []
    stem_pos = []
    sid = 0
    while sid < n_stems:
        x, y = rng.uniform(1, W - 1), rng.uniform(1, H - 1)
        if 7.0 < x < 8.4:  # canal
            continue
        if any(np.hypot(x - a, y - b) < 0.9 for a, b in stem_pos):
            continue
        sp = rng.choice(["R", "A", "L"], p=[0.55, 0.3, 0.15])
        stem_pos.append((x, y))
        sid += 1
        r0 = {"R": rng.uniform(0.04, 0.11), "A": rng.uniform(0.03, 0.10), "L": rng.uniform(0.025, 0.06)}[sp]
        height = rng.uniform(5.5, 10.0)
        lean = np.radians(rng.uniform(0, 8)); az = rng.uniform(0, 2 * np.pi)
        top = np.array([x + np.sin(lean) * np.cos(az) * height, y + np.sin(lean) * np.sin(az) * height, height])
        base = np.array([x, y, 0.0])
        h_root = rng.uniform(0.8, 2.4) if sp == "R" else 0.0
        pts_all, nrm_all = [], []
        pts, nrm = _cyl(rng, base, top, r0, dens); pts_all.append(pts); nrm_all.append(nrm)
        if sp == "R":
            for _ in range(rng.integers(6, 11)):
                th = rng.uniform(0, 2 * np.pi); d = rng.uniform(0.5, 1.0)
                f = h_root / height
                axis_pt = base + f * (top - base)
                p_ground = np.array([axis_pt[0] + d * np.cos(th), axis_pt[1] + d * np.sin(th), 0.0])
                pts, nrm = _cyl(rng, p_ground, axis_pt + r0 * np.array([np.cos(th), np.sin(th), 0]) * 0.9, rng.uniform(0.03, 0.06), dens)
                pts_all.append(pts); nrm_all.append(nrm)
        P = np.vstack(pts_all); N = np.vstack(nrm_all)
        allp.append((P, N))
        h_dbh = max(1.3, h_root + 0.15) if sp == "R" else 1.3
        f = h_dbh / height
        pos = base + f * (top - base)
        truth.append({"stem_id": sid, "species": sp, "x": pos[0], "y": pos[1], "dbh_cm": 200 * r0,
                      "dbh_height_m": h_dbh, "height_m": height, "h_root_m": h_root})
        # folhagem
        nl = int(300 * (1 if height > 4 else 0))
        leaf = top + rng.normal(0, [0.6, 0.6, 0.8], size=(nl, 3)); leaf[:, 2] = np.clip(leaf[:, 2], 3, None)
        allp.append((leaf, None))
    # solo
    gx = rng.uniform(0, W, int(W * H * 300)); gy = rng.uniform(0, H, len(gx))
    gz = 0.04 * np.sin(gx / 3) + 0.03 * np.cos(gy / 2) + rng.normal(0, 0.004, len(gx))
    ok = ~((gx > 7.0) & (gx < 8.4))  # canal sem retorno
    ground = np.column_stack([gx[ok], gy[ok], gz[ok]])
    mir = rng.uniform(0, 1, (400, 2)); mirror = np.column_stack([7.0 + 1.4 * mir[:, 0], mir[:, 1] * H, -0.5 + rng.normal(0, 0.05, 400)])
    understory = np.column_stack([rng.uniform(0, W, 1500), rng.uniform(0, H, 1500), rng.uniform(0.1, 1.2, 1500)])
    # visibilidade por scanner
    out = []
    for si, sc in enumerate(scanners):
        off = rng.normal(0, 0.004, 3) * np.array([1, 1, 0])
        for P, N in allp:
            if N is not None:
                v = sc - P
                rr = np.linalg.norm(v, axis=1)
                vis = ((v / rr[:, None]) * N).sum(1) > 0.1
                m = vis & (rr < 25) & (rng.random(len(P)) < 0.55)
            else:
                m = rng.random(len(P)) < 0.3
            Q = P[m] + off + rng.normal(0, 0.003, (m.sum(), 3))
            out.append(Q)
    cloud = np.vstack(out + [ground, mirror, understory])
    cloud = cloud[(cloud[:, 0] >= 0) & (cloud[:, 0] <= W) & (cloud[:, 1] >= 0) & (cloud[:, 1] <= H)]
    truth = pd.DataFrame(truth)
    if out_las:
        from .io_clouds import write_labeled_las
        write_labeled_las(out_las, cloud + np.array([500000.0, 7460000.0, 2.0]), np.full(len(cloud), -1), cloud[:, 2])
    return cloud, truth, W * H
