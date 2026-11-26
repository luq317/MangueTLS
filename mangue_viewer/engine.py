"""Logica da janela sem nada de interface: estado da parcela, recorte, solo, troncos, medicao manual e resumo.
Usa o pacote tls_mangue (Pipeline C) como motor."""
import math, time, json
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from tls_mangue.config import Params
from tls_mangue.io_clouds import read_cloud, thin_voxel
from tls_mangue.preprocess import preprocess
from tls_mangue.ground import build_dtm, normalize_height
from tls_mangue.stems import detect_stems, _stem_height

SPECIES = ["?", "Rh", "Av", "Lg"]
CONDS = ["?", "v", "m"]  # ? = nao classificado, v = viva, m = morta


def rect_corners(cx, cy, w, h, ang_deg):
    a = math.radians(ang_deg); c, s = math.cos(a), math.sin(a)
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([cx, cy])


def fit_circle(xy, iters=4):
    """Circulo por minimos quadrados com descarte iterativo de pontos fora (robusto a ruido). Retorna (cx, cy, r, rmse, arc_deg, n)."""
    xy = np.asarray(xy, float)
    if len(xy) < 8:
        return None
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        P = xy[keep]
        if len(P) < 8:
            return None
        A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
        b = (P ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-12))
        d = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        mad = np.median(d[keep]) + 1e-4
        keep = d < max(0.012, 3.0 * mad)
    P = xy[keep]
    d = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    ang = np.degrees(np.arctan2(P[:, 1] - cy, P[:, 0] - cx)) % 360
    arc = 10.0 * len(np.unique((ang // 10).astype(int)))
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(d ** 2))), float(arc), int(keep.sum())


class Parcela:
    def __init__(self):
        # padroes de teste em manguezal: nuvem a 2 cm, cadeia minima de 5 fatias, arco minimo 90 graus (mesmos dos testes em F2, H2, I2 e F3)
        self.p = Params(voxel_size=0.02, min_chain_slices=5, min_arc_deg=90.0, ground_regional_tol=0.6)
        self.path = None
        self.origin = np.zeros(3)
        self.raw = None          # (N,3) local (origem subtraida)
        self.disp = None         # subamostra para a tela
        self.crop = dict(cx=0.0, cy=0.0, w=15.0, h=15.0, ang=0.0)
        self.xyz_n = None        # dentro do recorte, z = altura acima do solo
        self.xyz_ground = None   # mesmo recorte com z original
        self.dtm = None
        self.report = {}
        self.stems = []          # lista de dicts
        self.log = []            # correcoes do usuario

    # ---------- 1. carregar ----------
    def carregar(self, path, thin=0.02, n_tela=3_000_000):
        self.path = str(path)
        raw = read_cloud(path, thin=thin if str(path).lower().endswith(".e57") else None)  # float64
        self.origin = raw.min(axis=0)                      # subtrai a origem em float64 ANTES de ir para float32 (UTM perde precisao em float32)
        loc = (raw - self.origin).astype(np.float32)
        del raw
        if not str(path).lower().endswith(".e57"):
            loc = thin_voxel(loc, thin)
        self.raw = loc
        self.disp = self._subamostra(self.raw, n_tela)
        c = self.raw[:, :2]
        self.crop = dict(cx=float((c[:, 0].min() + c[:, 0].max()) / 2), cy=float((c[:, 1].min() + c[:, 1].max()) / 2), w=15.0, h=15.0, ang=0.0)
        return dict(n=len(self.raw), n_tela=len(self.disp), extent=(self.raw.max(0) - self.raw.min(0)).tolist())

    @staticmethod
    def _subamostra(xyz, n_max):
        if len(xyz) <= n_max:
            return xyz
        rng = np.random.default_rng(0)
        return xyz[np.sort(rng.choice(len(xyz), n_max, replace=False))]

    # ---------- 2. recorte ----------
    def corners(self):
        c = self.crop
        return rect_corners(c["cx"], c["cy"], c["w"], c["h"], c["ang"])

    def area(self):
        return float(self.crop["w"] * self.crop["h"])

    def mask(self, xyz, margin=0.0):
        c = self.crop
        a = math.radians(-c["ang"]); cs, sn = math.cos(a), math.sin(a)
        dx, dy = xyz[:, 0] - c["cx"], xyz[:, 1] - c["cy"]
        u, v = dx * cs - dy * sn, dx * sn + dy * cs
        return (np.abs(u) <= c["w"] / 2 + margin) & (np.abs(v) <= c["h"] / 2 + margin)

    def sugerir_recorte(self, area_m2=225.0, quadrado=True):
        """Janela de area dada que contem mais pontos em altura (>0,5 m acima do 5o percentil de z), buscando posicao e angulo."""
        xyz = self.disp
        z0 = np.percentile(xyz[:, 2], 5)
        up = xyz[xyz[:, 2] > z0 + 0.5]
        if len(up) < 100:
            return self.crop
        L = math.sqrt(area_m2)
        best = None
        cx0, cy0 = np.median(up[:, 0]), np.median(up[:, 1])
        for ang in range(0, 90, 5):
            a = math.radians(-ang); cs, sn = math.cos(a), math.sin(a)
            u = (up[:, 0] - cx0) * cs - (up[:, 1] - cy0) * sn
            v = (up[:, 0] - cx0) * sn + (up[:, 1] - cy0) * cs
            H, ue, ve = np.histogram2d(u, v, bins=[np.arange(u.min(), u.max() + 0.5, 0.5), np.arange(v.min(), v.max() + 0.5, 0.5)])
            k = max(1, int(round(L / 0.5)))
            cs2 = H.cumsum(0).cumsum(1)
            cs2 = np.pad(cs2, ((1, 0), (1, 0)))
            if cs2.shape[0] <= k or cs2.shape[1] <= k:
                continue
            S = cs2[k:, k:] - cs2[:-k, k:] - cs2[k:, :-k] + cs2[:-k, :-k]
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if best is None or S[i, j] > best[0]:
                uc, vc = ue[i] + k * 0.25, ve[j] + k * 0.25
                a2 = math.radians(ang)
                cx = cx0 + uc * math.cos(a2) - vc * math.sin(a2)
                cy = cy0 + uc * math.sin(a2) + vc * math.cos(a2)
                best = (S[i, j], cx, cy, ang)
        _, cx, cy, ang = best
        self.crop = dict(cx=float(cx), cy=float(cy), w=L, h=L, ang=float(ang))
        return self.crop

    # ---------- 3. solo e altura ----------
    def solo(self, progress=None):
        p = self.p
        m = self.mask(self.raw, margin=1.0)
        xyz = self.raw[m].astype(np.float64)
        if len(xyz) < 1000:
            raise RuntimeError("Poucos pontos dentro do recorte. Ajuste a parcela.")
        xyz, rep_pre = preprocess(xyz, p)
        dtm, rep_g, _ = build_dtm(xyz, p)
        zn, keep = normalize_height(xyz, dtm, p)
        xyz, zn = xyz[keep], zn[keep]
        self.xyz_ground = xyz
        self.xyz_n = np.column_stack([xyz[:, 0], xyz[:, 1], zn])
        self.dtm = dtm
        self.report = dict(pre=rep_pre, ground=rep_g)
        return self.report

    def mdt_malha(self, step=0.25):
        c = self.corners()
        xs = np.arange(c[:, 0].min(), c[:, 0].max() + step, step); ys = np.arange(c[:, 1].min(), c[:, 1].max() + step, step)
        X, Y = np.meshgrid(xs, ys)
        Z = self.dtm(X.ravel(), Y.ravel()).reshape(X.shape)
        dentro = self.mask(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])).reshape(X.shape)
        Z = np.where(dentro, Z, np.nan)  # recorte girado: nao desenha o solo extrapolado fora da parcela
        return X, Y, Z

    # ---------- 4. troncos ----------
    def detectar(self):
        m = self.mask(self.xyz_n, margin=0.5)
        stems, rep = detect_stems(self.xyz_n[m], self.p)
        keep = []
        for s in stems:
            inside = self.mask(np.array([[s["x"], s["y"], 0.0]]))[0]
            if inside:
                keep.append(s)
        self.stems = []
        for i, s in enumerate(keep, 1):
            s = dict(s); s["stem_id"] = i; s.update(sp="?", cond="?", deleted=False, origem="auto")
            self.stems.append(s)
        self.log.append(dict(t=time.strftime("%H:%M:%S"), acao="detectar", n=len(self.stems)))
        return rep

    def _axis_xy(self, s, z):
        return float(np.polyval(s["_bx"], z)), float(np.polyval(s["_by"], z))

    def remedir(self, sid, h):
        """Refaz o DAP do fuste a altura h (m) com as fatias de +-5 cm em torno do eixo."""
        s = self.stems[sid - 1]
        cx0, cy0 = self._axis_xy(s, h) if s["origem"] == "auto" else (s["x"], s["y"])
        P = self.xyz_n
        rmax = max(0.35, 2.2 * s["_r"] + 0.1)
        m = (np.abs(P[:, 2] - h) <= 0.05) & (np.hypot(P[:, 0] - cx0, P[:, 1] - cy0) <= rmax)
        fit = fit_circle(P[m][:, :2])
        if fit is None or not (0.012 <= fit[2] <= 0.45) or fit[4] < 90:
            return None
        cx, cy, r, rmse, arc, n = fit
        old = s["dbh_cm"]
        s.update(x=cx, y=cy, dbh_cm=200 * r, dbh_height_m=float(h), basal_area_m2=math.pi * r ** 2, circle_rmse_cm=100 * rmse, arc_deg=arc, _r=r)
        if s["origem"] == "auto":
            s["origem"] = "auto+remedido"
        drop = {"REVIEW", "NO_SLICE_AT_DBH", "DBH_EXTRAPOLATED", "HIGH_DBH_HEIGHT", "ROOT_ADJUSTED"}
        s["flags"] = ";".join(f for f in s["flags"].split(";") if f and f not in drop)
        s["quality"] = max(s["quality"], 0.6)
        self.log.append(dict(t=time.strftime("%H:%M:%S"), acao="remedir", id=sid, h=h, dbh_antes=round(old, 2), dbh_depois=round(200 * r, 2)))
        return dict(dbh=200 * r, rmse_cm=100 * rmse, arc=arc)

    def adicionar(self, x, y, h=1.3, dbh_manual=None):
        P = self.xyz_n
        m = (np.abs(P[:, 2] - h) <= 0.05) & (np.hypot(P[:, 0] - x, P[:, 1] - y) <= 0.35)
        fit = fit_circle(P[m][:, :2])
        if dbh_manual is not None:
            cx, cy, r, rmse, arc, n = x, y, dbh_manual / 200.0, float("nan"), float("nan"), 0
        elif fit is None or not (0.012 <= fit[2] <= 0.45) or fit[4] < 90:
            return None
        else:
            cx, cy, r, rmse, arc, n = fit
        height, hocc = _stem_height(P, [cx], [cy], h, h, self.p)
        sid = len(self.stems) + 1
        s = dict(plot="", x=cx, y=cy, dbh_cm=200 * r, dbh_height_m=float(h), root_zone_top_m=float("nan"), height_m=float(height), height_occ_m=float(hocc),
                 basal_area_m2=math.pi * r ** 2, quality=0.7 if dbh_manual is None else 0.5, n_slices=1, n_slices_dbh=1, z_min=h, z_max=h, lean_deg=0.0,
                 circle_rmse_cm=100 * rmse if rmse == rmse else float("nan"), arc_deg=arc, radius_cv=float("nan"), flags="MANUAL", stem_id=sid,
                 _bx=[cx], _by=[cy], _r=r, sp="?", cond="?", deleted=False, origem="manual")
        self.stems.append(s)
        self.log.append(dict(t=time.strftime("%H:%M:%S"), acao="adicionar", id=sid, x=round(cx, 2), y=round(cy, 2), dbh=round(200 * r, 2)))
        return s

    def set_attr(self, sid, **kw):
        s = self.stems[sid - 1]
        for k, v in kw.items():
            if s.get(k) != v:
                self.log.append(dict(t=time.strftime("%H:%M:%S"), acao=k, id=sid, de=s.get(k), para=v))
                s[k] = v

    def proximo(self, x, y, tol=0.5):
        best, bd = None, tol
        for s in self.stems:
            d = math.hypot(s["x"] - x, s["y"] - y)
            if d < bd:
                best, bd = s["stem_id"], d
        return best

    # ---------- 5. resumo e exportacao ----------
    def tabela(self):
        cols = ["stem_id", "sp", "cond", "dbh_cm", "dbh_height_m", "height_m", "basal_area_m2", "quality", "flags", "origem", "deleted", "x", "y", "circle_rmse_cm", "arc_deg"]
        df = pd.DataFrame([{k: s.get(k) for k in cols} for s in self.stems], columns=cols)
        if len(df):
            df["x_utm"] = df.x + self.origin[0]; df["y_utm"] = df.y + self.origin[1]
        return df

    def resumo(self):
        """Regra do MangueTREE: cada tronco e um fuste; DAP medio = fuste de area basal media; altura = media das alturas."""
        df = self.tabela()
        A = self.area()
        out = {}
        for nome, sel in [("todos (nao removidos)", ~df.deleted), ("vivos + nao classificados", ~df.deleted & (df.cond != "m")), ("so vivos marcados", ~df.deleted & (df.cond == "v"))]:
            d = df[sel] if len(df) else df
            n = len(d)
            ab = float(np.sum(0.00007854 * d.dbh_cm ** 2)) if n else 0.0
            out[nome] = dict(n=n, dens_ha=n * 1e4 / A, ab_m2ha=ab * 1e4 / A, dap_cm=math.sqrt(ab * 12732.39 / n) if n else float("nan"),
                             h_m=float(d.height_m.mean()) if n else float("nan"))
        return out

    def exportar(self, pasta):
        import pathlib
        pasta = pathlib.Path(pasta); pasta.mkdir(parents=True, exist_ok=True)
        df = self.tabela(); df.to_csv(pasta / "arvores_corrigidas.csv", index=False, float_format="%.4f")
        pd.DataFrame(self.resumo()).T.to_csv(pasta / "resumo_parcela.csv", float_format="%.4f")
        with open(pasta / "correcoes.json", "w", encoding="utf-8") as f:
            json.dump(dict(nuvem=self.path, origem_xyz=self.origin.tolist(), recorte=self.crop, area_m2=self.area(), cantos_utm=(self.corners() + self.origin[:2]).tolist(),
                           params=self.p.__dict__, correcoes=self.log), f, indent=2, ensure_ascii=False, default=float)
        return str(pasta)
