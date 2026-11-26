"""Janela passo a passo (PROTOTIPO). Rodar: python iniciar.py [nuvem.laz|.e57]"""
import sys, os, time, traceback
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor

from .engine import Parcela, SPECIES, CONDS, rect_corners

BG1, BG2 = "#0b1a2b", "#16314d"
TEAL, NAVY = "#2bb3a3", "#0b2a4a"
SP_COL = {"Rh": (43, 179, 163), "Av": (240, 180, 60), "Lg": (214, 98, 130), "?": (150, 160, 170)}
COND_COL = {"v": (80, 190, 110), "m": (170, 90, 60), "?": (150, 160, 170)}
STEPS = ["1  Carregar nuvem", "2  Recortar parcela", "3  Solo e altura", "4  Árvores e DAP", "5  Resumo e exportar"]

QSS = f"""
QMainWindow, QWidget {{ background: #0f1c2b; color: #dbe6ef; font-family: 'Segoe UI','SF Pro Text','Helvetica Neue',Arial; font-size: 13px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 11px 12px; margin: 2px 0; border-radius: 8px; color: #8fa6bb; }}
QListWidget::item:selected {{ background: {NAVY}; color: white; border-left: 3px solid {TEAL}; }}
QPushButton {{ background: #1b3550; border: 1px solid #27486a; border-radius: 8px; padding: 8px 14px; color: #e8f1f8; }}
QPushButton:hover {{ background: #24476a; }}
QPushButton:disabled {{ color: #5d7389; background: #142538; }}
QPushButton#primary {{ background: {TEAL}; border: none; color: #04221e; font-weight: 600; }}
QPushButton#primary:hover {{ background: #3cc7b6; }}
QPushButton#primary:disabled {{ background: #1c5a54; color: #7aa7a2; }}
QLabel#h {{ font-size: 15px; font-weight: 600; color: white; }}
QLabel#muted {{ color: #8fa6bb; }}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{ background: #132a40; border: 1px solid #27486a; border-radius: 6px; padding: 4px 6px; color: white; }}
QTableWidget {{ background: #0e2033; gridline-color: #1d3550; border: 1px solid #1d3550; border-radius: 6px; selection-background-color: {NAVY}; }}
QHeaderView::section {{ background: #132a40; color: #9bb3c8; border: none; padding: 4px; }}
QProgressBar {{ background: #132a40; border: none; border-radius: 4px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 4px; }}
QCheckBox {{ spacing: 6px; }}
"""


class Worker(QtCore.QThread):
    done = QtCore.Signal(object, object)

    def __init__(self, fn, *a, **k):
        super().__init__(); self.fn, self.a, self.k = fn, a, k

    def run(self):
        try:
            self.done.emit(self.fn(*self.a, **self.k), None)
        except Exception as e:  # mostra o erro na janela em vez de fechar o programa
            self.done.emit(None, "".join(traceback.format_exception_only(type(e), e)).strip())


def lab(text, name=None):
    l = QtWidgets.QLabel(text); l.setWordWrap(True)
    if name: l.setObjectName(name)
    return l


def spin(lo, hi, val, step=0.5, dec=2, suf=""):
    s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
    if suf: s.setSuffix(suf)
    return s


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MangueTLS Viewer (protótipo de teste)")
        self.resize(1480, 880)
        self.par = Parcela()
        self.step = 0
        self.sel = None
        self.mode_add = False
        self.worker = None
        self.maxstep = 0
        self._build()
        self.go(0)

    # ------------------------------------------------------------ interface
    def _build(self):
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        h = QtWidgets.QHBoxLayout(root); h.setContentsMargins(12, 12, 12, 8); h.setSpacing(12)
        # esquerda
        left = QtWidgets.QVBoxLayout(); left.setSpacing(8)
        t = QtWidgets.QLabel("MangueTLS"); t.setStyleSheet(f"font-size:20px;font-weight:700;color:{TEAL};")
        left.addWidget(t); left.addWidget(lab("Parâmetros estruturais de manguezal a partir de nuvens TLS", "muted"))
        self.steps = QtWidgets.QListWidget(); self.steps.setFixedWidth(210)
        for s in STEPS: self.steps.addItem(s)
        self.steps.currentRowChanged.connect(self._clicou_passo)
        left.addWidget(self.steps, 1)
        nav = QtWidgets.QHBoxLayout()
        self.b_back = QtWidgets.QPushButton("Voltar"); self.b_next = QtWidgets.QPushButton("Avançar"); self.b_next.setObjectName("primary")
        self.b_back.clicked.connect(lambda: self.go(self.step - 1)); self.b_next.clicked.connect(lambda: self.go(self.step + 1))
        nav.addWidget(self.b_back); nav.addWidget(self.b_next); left.addLayout(nav)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(220); h.addWidget(lw)
        # centro
        mid = QtWidgets.QVBoxLayout(); mid.setSpacing(6)
        self.pl = QtInteractor(self); self.pl.set_background(BG1, top=BG2)
        mid.addWidget(self.pl.interactor, 1)
        bar = QtWidgets.QHBoxLayout()
        for txt, fn in [("Planta", self.view_top), ("Lateral", self.view_side), ("3D", self.view_iso)]:
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); bar.addWidget(b)
        self.c_labels = QtWidgets.QCheckBox("Números das árvores"); self.c_labels.setChecked(True); self.c_labels.toggled.connect(self.draw_trees)
        self.c_edl = QtWidgets.QCheckBox("Sombreado de profundidade"); self.c_edl.setChecked(True); self.c_edl.toggled.connect(self._edl)
        bar.addWidget(lab("Nuvem até")); self.s_zmax = spin(1.0, 30.0, 5.0, 0.5, 1, " m"); self.s_zmax.setToolTip("Nos passos 4 e 5 mostra só a nuvem entre 0,4 m e essa altura acima do solo, para ver os troncos sem o chão nem o dossel. 30 m mostra tudo acima de 0,4 m."); self.s_zmax.valueChanged.connect(lambda *_: self.render()); bar.addWidget(self.s_zmax)
        bar.addStretch(1); bar.addWidget(self.c_labels); bar.addWidget(self.c_edl)
        mid.addLayout(bar)
        self.prog = QtWidgets.QProgressBar(); self.prog.setRange(0, 1); self.prog.setValue(0); mid.addWidget(self.prog)
        self.status = lab("Abra uma nuvem para começar.", "muted"); mid.addWidget(self.status)
        mw = QtWidgets.QWidget(); mw.setLayout(mid); h.addWidget(mw, 1)
        # direita
        self.stack = QtWidgets.QStackedWidget(); self.stack.setFixedWidth(380)
        self.stack.addWidget(self._p_carregar()); self.stack.addWidget(self._p_recorte()); self.stack.addWidget(self._p_solo())
        self.stack.addWidget(self._p_arvores()); self.stack.addWidget(self._p_resumo())
        h.addWidget(self.stack)
        self._edl(True)
        try:
            self.pl.enable_point_picking(callback=self._pick, left_clicking=True, show_message=False, show_point=False, use_picker=False)
        except Exception as e:
            print("picking indisponivel:", e)

    def _panel(self, titulo, texto):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(4, 0, 0, 0); l.setSpacing(10)
        l.addWidget(lab(titulo, "h")); l.addWidget(lab(texto, "muted"))
        return w, l

    def _p_carregar(self):
        w, l = self._panel("Carregar a nuvem", "Abra o arquivo exportado do FARO SCENE ou do ReCap (.e57, .laz, .las, .xyz, .pts). A nuvem é reduzida para 2 cm na leitura e uma amostra é mostrada na tela.")
        self.b_open = QtWidgets.QPushButton("Abrir nuvem..."); self.b_open.setObjectName("primary"); self.b_open.clicked.connect(self.abrir)
        l.addWidget(self.b_open)
        self.s_tela = spin(0.5, 10, 3.0, 0.5, 1, " milhões de pontos na tela"); l.addWidget(self.s_tela)
        self.i_carregar = lab(""); l.addWidget(self.i_carregar); l.addStretch(1)
        return w

    def _p_recorte(self):
        w, l = self._panel("Recortar a parcela", "Ajuste o retângulo para coincidir com a parcela. A área nominal define a densidade e a área basal por hectare. Pontos fora ficam em cinza.")
        g = QtWidgets.QFormLayout()
        self.s_cx = spin(-1e5, 1e5, 0, 0.5); self.s_cy = spin(-1e5, 1e5, 0, 0.5)
        self.s_w = spin(1, 200, 15, 0.5, 2, " m"); self.s_h = spin(1, 200, 15, 0.5, 2, " m"); self.s_ang = spin(-180, 180, 0, 1, 1, "°")
        for t, s in [("Centro X", self.s_cx), ("Centro Y", self.s_cy), ("Largura", self.s_w), ("Altura", self.s_h), ("Ângulo", self.s_ang)]:
            g.addRow(t, s); s.valueChanged.connect(self._crop_mudou)
        l.addLayout(g)
        self.l_area = lab("", "h"); l.addWidget(self.l_area)
        r = QtWidgets.QHBoxLayout(); self.cb_area = QtWidgets.QComboBox(); self.cb_area.addItems(["80 m²", "195 m²", "225 m²", "240 m²", "315 m²"]); self.cb_area.setCurrentIndex(2); r.addWidget(self.cb_area)
        b = QtWidgets.QPushButton("Sugerir posição"); b.clicked.connect(self.sugerir); r.addWidget(b); l.addLayout(r)
        l.addWidget(lab("A sugestão busca o quadrado dessa área que contém mais vegetação. Confira e ajuste à mão.", "muted"))
        l.addStretch(1)
        return w

    def _p_solo(self):
        w, l = self._panel("Solo e altura", "Remove ruído, estima o terreno por percentil (robusto à água e às raízes) e calcula a altura acima do solo. Células sem retorno viram lacuna e são preenchidas por interpolação.")
        self.b_solo = QtWidgets.QPushButton("Calcular solo e altura"); self.b_solo.setObjectName("primary"); self.b_solo.clicked.connect(self.run_solo); l.addWidget(self.b_solo)
        self.i_solo = lab(""); l.addWidget(self.i_solo); l.addStretch(1)
        return w

    def _p_arvores(self):
        w, l = self._panel("Árvores e DAP", "Clique numa árvore na nuvem ou na tabela para ver como foi medida. O anel claro marca a altura do DAP. Corrija o que estiver errado.")
        self.b_det = QtWidgets.QPushButton("Detectar árvores"); self.b_det.setObjectName("primary"); self.b_det.clicked.connect(self.run_detect); l.addWidget(self.b_det)
        r = QtWidgets.QHBoxLayout(); r.addWidget(lab("Colorir por")); self.cb_col = QtWidgets.QComboBox(); self.cb_col.addItems(["Qualidade", "Espécie", "Vivo ou morta", "DAP"])
        self.cb_col.currentIndexChanged.connect(self.draw_trees); r.addWidget(self.cb_col, 1)
        self.c_rev = QtWidgets.QCheckBox("Só para revisar"); self.c_rev.toggled.connect(self.fill_table); r.addWidget(self.c_rev); l.addLayout(r)
        self.tab = QtWidgets.QTableWidget(0, 7); self.tab.setHorizontalHeaderLabels(["ID", "Esp", "Cond", "DAP", "Alt", "Q", "Flags"])
        self.tab.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self.tab.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tab.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers); self.tab.verticalHeader().setVisible(False)
        self.tab.horizontalHeader().setStretchLastSection(True); self.tab.itemSelectionChanged.connect(self._tab_sel)
        for i, wd in enumerate([40, 40, 40, 50, 46, 40]): self.tab.setColumnWidth(i, wd)
        l.addWidget(self.tab, 1)
        self.l_sel = lab("Nenhuma árvore selecionada.", "muted"); l.addWidget(self.l_sel)
        g = QtWidgets.QGridLayout()
        self.cb_sp = QtWidgets.QComboBox(); self.cb_sp.addItems(SPECIES); self.cb_cd = QtWidgets.QComboBox(); self.cb_cd.addItems(["? sem classe", "v viva", "m morta"])
        self.cb_sp.activated.connect(self._set_sp); self.cb_cd.activated.connect(self._set_cd)
        g.addWidget(lab("Espécie"), 0, 0); g.addWidget(self.cb_sp, 0, 1); g.addWidget(lab("Condição"), 0, 2); g.addWidget(self.cb_cd, 0, 3)
        self.s_hm = spin(0.3, 6.0, 1.3, 0.1, 2, " m"); b1 = QtWidgets.QPushButton("Remedir DAP nessa altura"); b1.clicked.connect(self.remedir)
        g.addWidget(self.s_hm, 1, 0, 1, 2); g.addWidget(b1, 1, 2, 1, 2)
        self.b_del = QtWidgets.QPushButton("Remover / restaurar"); self.b_del.clicked.connect(self.remover)
        self.b_add = QtWidgets.QPushButton("Adicionar árvore (clique)"); self.b_add.setCheckable(True); self.b_add.toggled.connect(self._modo_add)
        g.addWidget(self.b_del, 2, 0, 1, 2); g.addWidget(self.b_add, 2, 2, 1, 2); l.addLayout(g)
        return w

    def _p_resumo(self):
        w, l = self._panel("Resumo da parcela", "Regra do MangueTREE: cada tronco é um fuste, DAP médio é o do fuste de área basal média, altura é a média das alturas. Árvores removidas não entram.")
        self.t_res = QtWidgets.QTableWidget(5, 3); self.t_res.setHorizontalHeaderLabels(["Todos", "Vivos+?", "Só vivos"]); self.t_res.setVerticalHeaderLabels(["Troncos", "Dens. (/ha)", "AB (m²/ha)", "DAP (cm)", "Altura (m)"])
        self.t_res.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers); self.t_res.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); self.t_res.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.t_res.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.t_res.verticalHeader().setDefaultSectionSize(30); self.t_res.setFixedHeight(200)
        l.addWidget(self.t_res)
        self.b_exp = QtWidgets.QPushButton("Exportar resultados..."); self.b_exp.setObjectName("primary"); self.b_exp.clicked.connect(self.exportar); l.addWidget(self.b_exp)
        self.i_exp = lab(""); l.addWidget(self.i_exp); l.addStretch(1)
        return w

    # ------------------------------------------------------------ navegação
    def go(self, k):
        k = max(0, min(len(STEPS) - 1, k))
        if k > self.maxstep + 0 and k > self.step + 1: return
        if k > 0 and self.par.raw is None: self.msg("Abra uma nuvem primeiro."); return
        if k >= 3 and self.par.xyz_n is None: self.msg("Calcule o solo antes (passo 3)."); return
        if k >= 4 and not self.par.stems: self.msg("Detecte as árvores antes (passo 4)."); return
        self.step = k; self.maxstep = max(self.maxstep, k)
        self.steps.blockSignals(True); self.steps.setCurrentRow(k); self.steps.blockSignals(False)
        self.stack.setCurrentIndex(k); self.b_back.setEnabled(k > 0); self.b_next.setEnabled(k < len(STEPS) - 1)
        if k == 4: self.render_resumo()
        self.render()
        if k == 4: self.view_iso()

    def _clicou_passo(self, r):
        if r >= 0 and r != self.step: self.go(r)

    def msg(self, t): self.status.setText(t)

    def busy(self, on, texto=""):
        self.prog.setRange(0, 0 if on else 1); self.prog.setValue(0)
        for b in (self.b_open, self.b_solo, self.b_det, self.b_next, self.b_back): b.setEnabled(not on)
        if texto: self.msg(texto)

    def roda(self, fn, ok, texto):
        self.busy(True, texto)
        self.worker = Worker(fn)
        def fim(res, err):
            self.busy(False); self.go(self.step)
            if err: self.msg("Erro: " + err); return
            ok(res)
        self.worker.done.connect(fim); self.worker.start()

    # ------------------------------------------------------------ passos
    def abrir(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Abrir nuvem", "", "Nuvens (*.e57 *.laz *.las *.xyz *.pts *.txt *.csv)")
        if path: self.carregar(path)

    def carregar(self, path):
        n_tela = int(self.s_tela.value() * 1e6)
        def ok(info):
            self.i_carregar.setText(f"{os.path.basename(path)}\n{info['n']:,} pontos após redução, {info['n_tela']:,} na tela.\nExtensão: {info['extent'][0]:.1f} × {info['extent'][1]:.1f} × {info['extent'][2]:.1f} m")
            self._sync_crop(); self.msg("Nuvem carregada. Avance para recortar a parcela."); self.render(reset=True)
        self.roda(lambda: self.par.carregar(path, n_tela=n_tela), ok, "Lendo e reduzindo a nuvem... (arquivos E57 grandes levam alguns minutos)")

    def _sync_crop(self):
        c = self.par.crop
        for s, k in [(self.s_cx, "cx"), (self.s_cy, "cy"), (self.s_w, "w"), (self.s_h, "h"), (self.s_ang, "ang")]:
            s.blockSignals(True); s.setValue(c[k]); s.blockSignals(False)
        self.l_area.setText(f"Área: {self.par.area():.1f} m²")

    def _crop_mudou(self):
        self.par.crop = dict(cx=self.s_cx.value(), cy=self.s_cy.value(), w=self.s_w.value(), h=self.s_h.value(), ang=self.s_ang.value())
        self.par.xyz_n = None; self.par.stems = []
        self.l_area.setText(f"Área: {self.par.area():.1f} m²"); self.render()

    def sugerir(self):
        if self.par.raw is None: return
        a = float(self.cb_area.currentText().split()[0])
        self.par.sugerir_recorte(a); self._sync_crop(); self._crop_mudou()

    def run_solo(self):
        def ok(rep):
            g = rep["ground"]
            self.i_solo.setText(f"Pontos após o filtro de ruído: {len(self.par.xyz_n):,}\nSolo: {g}")
            self.msg("Solo calculado. Avance para detectar as árvores."); self.render(reset=True)
        self.roda(self.par.solo, ok, "Filtrando ruído, estimando o solo e normalizando a altura...")

    def run_detect(self):
        def ok(rep):
            self.sel = None; self.fill_table(); self.msg(f"{len(self.par.stems)} árvores detectadas. Clique para conferir e corrigir."); self.render(reset=False)
        self.roda(self.par.detectar, ok, "Detectando troncos em várias alturas e medindo o DAP...")

    # ------------------------------------------------------------ desenho 3D
    def _edl(self, on):
        try:
            if on: self.pl.enable_eye_dome_lighting()
            else: self.pl.disable_eye_dome_lighting()
        except Exception as e: print("EDL indisponivel:", e)

    def _cloud_mesh(self, xyz, scal):
        m = pv.PolyData(np.ascontiguousarray(xyz, dtype=np.float32)); m["h"] = scal; return m

    def render(self, reset=False):
        pl = self.pl; par = self.par
        for n in ("fora", "dentro", "mdt", "retangulo", "arvores", "aneis", "sel", "rotulos"):
            try: pl.remove_actor(n, render=False)
            except Exception: pass
        if par.raw is None: pl.render(); return
        if self.step <= 1 or par.xyz_n is None:
            d = par.disp; m = par.mask(d)
            if self.step == 0: m[:] = True
            z = d[:, 2]; lo, hi = np.percentile(z, 1), np.percentile(z, 99.5)
            if (~m).any(): pl.add_mesh(self._cloud_mesh(d[~m], z[~m]), color="#566879", point_size=1.6, name="fora", opacity=0.55, reset_camera=False)
            if m.any(): pl.add_mesh(self._cloud_mesh(d[m], z[m]), scalars="h", cmap="viridis", clim=[lo, hi], point_size=2.2, show_scalar_bar=False, name="dentro", reset_camera=False)
            if self.step >= 1: self._draw_rect(float(z.min()))
        else:
            P = par.xyz_n
            m = par.mask(P, margin=0.2)
            hn = P[m][:, 2]; P = P[m]            # hn = altura acima do solo (cor)
            zoff = 0.0
            if self.step == 2 and par.dtm is not None:
                P = par.xyz_ground[m]            # passo 3: nuvem na altitude original, no mesmo referencial da malha do solo
                zoff = float(np.median(par.mdt_malha()[2][np.isfinite(par.mdt_malha()[2])]))
            if self.step >= 3:
                k = (hn >= 0.4) & (hn <= self.s_zmax.value())  # esconde o chao e o dossel para ver os troncos
                P, hn = P[k], hn[k]
            if len(P) > 2_500_000:
                sel = np.random.default_rng(0).choice(len(P), 2_500_000, replace=False); P, hn = P[sel], hn[sel]
            hh = np.clip(hn, 0, 12)
            pl.add_mesh(self._cloud_mesh(P, hh), scalars="h", cmap="viridis", clim=[0, max(2.0, min(float(np.percentile(hh, 99)), self.s_zmax.value() if self.step >= 3 else 99))], point_size=2.2, show_scalar_bar=False, name="dentro", reset_camera=False, opacity=1.0 if self.step == 2 else 0.6)
            if self.step == 2 and par.dtm is not None:
                X, Y, Z = par.mdt_malha(); g = pv.StructuredGrid(X, Y, Z)
                pl.add_mesh(g, color="#8a6f4e", opacity=0.55, name="mdt", reset_camera=False)
            if self.step >= 3: self.draw_trees()
            self._draw_rect(zoff)
        if reset or self.step in (1,): (self.view_top if self.step in (1, 2) else self.view_iso)()
        pl.render()

    def _draw_rect(self, z):
        c = self.par.corners(); pts = np.column_stack([c, np.full(4, z)]); pts = np.vstack([pts, pts[:1]])
        pl = self.pl; pl.add_mesh(pv.lines_from_points(pts), color=TEAL, line_width=3, name="retangulo", reset_camera=False)

    def _cor(self, s, mode):
        if s["deleted"]: return (120, 60, 60)
        if mode == 1: return SP_COL[s["sp"]]
        if mode == 2: return COND_COL[s["cond"]]
        if mode == 3:
            v = np.clip((s["dbh_cm"] - 3) / 30, 0, 1); return tuple(int(x) for x in (40 + 200 * v, 140 + 60 * (1 - abs(v - .5) * 2), 220 - 190 * v))
        q = np.clip(s["quality"], 0, 1); return (int(230 * (1 - q) + 40 * q), int(80 * (1 - q) + 190 * q), int(70 * (1 - q) + 120 * q))

    def draw_trees(self, *_):
        pl = self.pl; par = self.par
        for n in ("arvores", "aneis", "sel", "rotulos"):
            try: pl.remove_actor(n, render=False)
            except Exception: pass
        if self.step < 3 or not par.stems: pl.render(); return
        mode = self.cb_col.currentIndex()
        tubes, rings, lp, ll = [], [], [], []
        for s in par.stems:
            ztop = s["height_m"] if np.isfinite(s["height_m"]) else s["z_max"]
            zs = np.array([0.0, max(ztop, 1.0) / 2, max(ztop, 1.0)])
            pts = np.column_stack([np.polyval(s["_bx"], zs) * np.ones(3), np.polyval(s["_by"], zs) * np.ones(3), zs])
            r = max(s["_r"], 0.025)
            t = pv.Spline(pts, 8).tube(radius=r, n_sides=10)
            t["rgb"] = np.tile(np.array(self._cor(s, mode), np.uint8), (t.n_points, 1)); tubes.append(t)
            ring = pv.Cylinder(center=(s["x"], s["y"], s["dbh_height_m"]), direction=(0, 0, 1), radius=r * 1.35, height=0.05, resolution=24, capping=False)
            ring["rgb"] = np.tile(np.array((250, 250, 250), np.uint8), (ring.n_points, 1)); rings.append(ring)
            lp.append([s["x"], s["y"], zs[-1] + 0.25]); ll.append(str(s["stem_id"]))
        if tubes: pl.add_mesh(tubes[0].merge(tubes[1:]) if len(tubes) > 1 else tubes[0], scalars="rgb", rgb=True, name="arvores", reset_camera=False, pickable=True, smooth_shading=True)
        if rings: pl.add_mesh(rings[0].merge(rings[1:]) if len(rings) > 1 else rings[0], scalars="rgb", rgb=True, name="aneis", reset_camera=False, pickable=False)
        if self.c_labels.isChecked() and lp:
            pl.add_point_labels(np.array(lp), ll, font_size=13, text_color="white", shape="rounded_rect", shape_color=NAVY, shape_opacity=0.75, margin=2, show_points=False, always_visible=True, name="rotulos", pickable=False, render=False)
        self._draw_sel(); pl.render()

    def _draw_sel(self):
        try: self.pl.remove_actor("sel", render=False)
        except Exception: pass
        if self.sel is None: return
        s = self.par.stems[self.sel - 1]
        pl = self.pl
        ring = pv.Cylinder(center=(s["x"], s["y"], s["dbh_height_m"]), direction=(0, 0, 1), radius=max(s["_r"], 0.025) * 2.4, height=0.03, resolution=36, capping=False)
        pl.add_mesh(ring, color="#ffd24a", name="sel", reset_camera=False, pickable=False, line_width=3)

    # ------------------------------------------------------------ câmera
    def _cam(self, pos, up):
        c = self.par.crop; pl = self.pl
        d = max(c["w"], c["h"]) * 1.15
        f = (c["cx"], c["cy"], 1.5 if self.step >= 2 else 3.0)
        pl.camera_position = [(f[0] + pos[0] * d, f[1] + pos[1] * d, f[2] + pos[2] * d), f, up]; pl.reset_camera_clipping_range(); pl.render()

    def view_top(self): self._cam((0.001, -0.001, 1.6), (0, 1, 0))
    def view_side(self): self._cam((0, -1.5, 0.12), (0, 0, 1))
    def view_iso(self): self._cam((0.9, -1.1, 0.8), (0, 0, 1))

    def _fly(self, s):
        pl = self.pl; f = (s["x"], s["y"], min(max(s["height_m"] / 2 if np.isfinite(s["height_m"]) else 2, 1.0), 4))
        c = self.par.crop; dx, dy = f[0] - c["cx"], f[1] - c["cy"]; n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        pos = (f[0] + dx / n * 5.0 - dy / n * 2.0, f[1] + dy / n * 5.0 + dx / n * 2.0, f[2] + 1.2)   # visto de fora da parcela, ligeiramente de lado
        pl.camera_position = [pos, f, (0, 0, 1)]; pl.reset_camera_clipping_range(); pl.render()

    # ------------------------------------------------------------ seleção e correção
    def fill_table(self, *_):
        t = self.tab; t.blockSignals(True); t.setRowCount(0)
        for s in self.par.stems:
            if self.c_rev.isChecked() and "REVIEW" not in s["flags"] and s["origem"] == "auto": continue
            r = t.rowCount(); t.insertRow(r)
            vals = [s["stem_id"], s["sp"], s["cond"], f"{s['dbh_cm']:.1f}", f"{s['height_m']:.1f}" if np.isfinite(s["height_m"]) else "-", f"{s['quality']:.2f}", ("(removida) " if s["deleted"] else "") + s["flags"]]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v)); it.setData(Qt.UserRole, s["stem_id"])
                if s["deleted"]: it.setForeground(QtGui.QColor("#a07070"))
                elif "REVIEW" in s["flags"]: it.setForeground(QtGui.QColor("#f2c14e"))
                t.setItem(r, c, it)
        t.blockSignals(False)

    def _tab_sel(self):
        r = self.tab.selectedItems()
        if r: self.select(int(r[0].data(Qt.UserRole)), fly=True, from_table=True)

    def select(self, sid, fly=False, from_table=False):
        self.sel = sid; s = self.par.stems[sid - 1]
        fl = s["flags"] or "sem alertas"
        self.l_sel.setText(f"Árvore {sid}: DAP {s['dbh_cm']:.1f} cm a {s['dbh_height_m']:.2f} m, altura {s['height_m']:.1f} m, qualidade {s['quality']:.2f}, arco {s['arc_deg']:.0f}°, origem {s['origem']}.\nAlertas: {fl}")
        self.cb_sp.setCurrentText(s["sp"]); self.cb_cd.setCurrentIndex(CONDS.index(s["cond"])); self.s_hm.setValue(float(s["dbh_height_m"]))
        if not from_table:
            for r in range(self.tab.rowCount()):
                if int(self.tab.item(r, 0).data(Qt.UserRole)) == sid:
                    self.tab.blockSignals(True); self.tab.selectRow(r); self.tab.blockSignals(False); self.tab.scrollToItem(self.tab.item(r, 0)); break
        self._draw_sel()
        if fly: self._fly(s)
        self.pl.render()

    def _pick(self, pt):
        if pt is None or self.step != 3 or not self.par.stems and not self.mode_add: return
        x, y = float(pt[0]), float(pt[1])
        if self.mode_add:
            s = self.par.adicionar(x, y, float(self.s_hm.value()))
            if s is None:
                v, ok = QtWidgets.QInputDialog.getDouble(self, "Adicionar árvore", "Não foi possível medir o círculo ali. Informe o DAP (cm) manualmente:", 8.0, 1.0, 80.0, 1)
                if not ok: return
                s = self.par.adicionar(x, y, float(self.s_hm.value()), dbh_manual=v)
            self.b_add.setChecked(False); self.fill_table(); self.draw_trees(); self.select(s["stem_id"]); return
        sid = self.par.proximo(x, y, 0.5)
        if sid: self.select(sid)

    def _modo_add(self, on):
        self.mode_add = on; self.msg("Clique na base do tronco que falta na nuvem." if on else "")

    def _atualiza(self, sid):
        self.fill_table(); self.draw_trees(); self.select(sid)

    def _set_sp(self, i):
        if self.sel: self.par.set_attr(self.sel, sp=SPECIES[i]); self._atualiza(self.sel)

    def _set_cd(self, i):
        if self.sel: self.par.set_attr(self.sel, cond=CONDS[i]); self._atualiza(self.sel)

    def remover(self):
        if self.sel: s = self.par.stems[self.sel - 1]; self.par.set_attr(self.sel, deleted=not s["deleted"]); self._atualiza(self.sel)

    def remedir(self):
        if not self.sel: return
        r = self.par.remedir(self.sel, float(self.s_hm.value()))
        if r is None:
            v, ok = QtWidgets.QInputDialog.getDouble(self, "Remedir", "Não há pontos suficientes nessa altura. Informe o DAP (cm) medido na nuvem:", self.par.stems[self.sel - 1]["dbh_cm"], 1.0, 80.0, 1)
            if ok:
                s = self.par.stems[self.sel - 1]; self.par.set_attr(self.sel, dbh_cm=v, _r=v / 200.0, basal_area_m2=np.pi * (v / 200.0) ** 2, dbh_height_m=float(self.s_hm.value()), origem="auto+manual")
            else: return
        self.msg("DAP atualizado."); self._atualiza(self.sel)

    # ------------------------------------------------------------ resumo
    def render_resumo(self):
        res = self.par.resumo()
        for c, (nome, v) in enumerate(res.items()):
            for r, val in enumerate([v["n"], v["dens_ha"], v["ab_m2ha"], v["dap_cm"], v["h_m"]]):
                it = QtWidgets.QTableWidgetItem("-" if val != val else (f"{val:.0f}" if r < 2 else f"{val:.2f}")); self.t_res.setItem(r, c, it)

    def exportar(self):
        pasta = QtWidgets.QFileDialog.getExistingDirectory(self, "Pasta de saída")
        if not pasta: return
        out = self.par.exportar(pasta)
        try: self.pl.screenshot(os.path.join(out, "cena.png"))
        except Exception: pass
        self.i_exp.setText(f"Salvo em {out}:\narvores_corrigidas.csv, resumo_parcela.csv, correcoes.json, cena.png")


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    app = QtWidgets.QApplication(argv); app.setStyleSheet(QSS)
    w = Main(); w.show()
    if len(argv) > 1 and os.path.exists(argv[1]): QtCore.QTimer.singleShot(300, lambda: w.carregar(argv[1]))
    return app.exec()
