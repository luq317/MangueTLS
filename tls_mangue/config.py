"""Parametros criticos do pipeline. Todos precisam ser calibrados com dados de manguezal."""
from dataclasses import dataclass, asdict, fields
import json


@dataclass
class Params:
    # ---- 1. Pre-processamento ----
    voxel_size: float = 0.01        # m. Subamostragem por voxel (1 a 2 cm para parcelas de 200 a 300 m2).
    sor_k: int = 8                  # vizinhos do SOR
    sor_std: float = 2.0            # desvios-padrao do SOR (menor remove mais)
    ror_radius: float = 0.10        # m. Raio do ROR
    ror_min_neighbors: int = 3      # vizinhos minimos dentro do raio do ROR

    # ---- 2. Solo e normalizacao de altura ----
    ground_cell: float = 0.5        # m. Celula da grade do solo
    ground_percentile: float = 5.0  # percentil baixo de z por celula (evita pneumatoforos e raizes)
    ground_min_points: int = 5      # pontos minimos por celula para a celula contar
    ground_spike_tol: float = 0.25  # m. Celulas que desviam da mediana local alem disso viram lacuna
    ground_regional_tol: float = 0.0  # m. Filtro regional do solo (0 = desligado; o viewer usa 0,6)
    ground_regional_cells: int = 9    # janela (celulas) da mediana regional
    ground_smooth_cells: int = 3    # suavizacao (em celulas) aplicada so as celulas preenchidas
    below_ground_cut: float = -0.30 # m. Pontos abaixo disso (reflexo na agua) sao descartados

    # ---- 3. Deteccao de fustes em varias fatias ----
    slice_min: float = 0.5          # m. Primeira fatia (altura acima do solo)
    slice_max: float = 4.0          # m. Ultima fatia
    slice_step: float = 0.25        # m. Passo entre fatias
    slice_thickness: float = 0.12   # m. Espessura de cada fatia
    cluster_eps: float = 0.05       # m. DBSCAN dentro da fatia
    cluster_min_samples: int = 8
    max_circles_per_cluster: int = 3
    r_min: float = 0.015            # m. Raio minimo do fuste (DAP minimo = 2 * r_min)
    r_max: float = 0.40             # m. Raio maximo
    ransac_iterations: int = 200
    ransac_tolerance: float = 0.010 # m. Distancia maxima ao circulo para ser inlier
    min_inliers: int = 12
    min_arc_deg: float = 120.0      # arco minimo visto do circulo (bins de 10 graus)
    max_circle_rmse: float = 0.015  # m
    max_inner_fraction: float = 0.25  # fracao maxima de pontos dentro de 0,6 r (rejeita blocos densos)

    # ---- 4. Ligacao entre fatias e raizes-escora ----
    link_dist: float = 0.15         # m. Deslocamento horizontal maximo entre fatias
    max_gap_slices: int = 2         # fatias consecutivas sem circulo toleradas
    min_chain_slices: int = 6       # fatias minimas por fuste
    max_lean_deg: float = 40.0      # inclinacao maxima do eixo
    root_attach_margin: float = 0.12  # m. Margem para considerar que uma cadeia encosta no tronco
    root_radius_ratio: float = 1.5  # o tronco precisa ter raio >= esta razao x o da cadeia candidata a raiz
    root_extrapolate: float = 1.5   # m. Quanto o eixo da cadeia candidata a raiz e prolongado para cima
    merge_radius: float = 0.12      # m. Cadeias mais proximas que isso sao duplicatas

    # ---- 5. DAP (altura de medicao adaptativa) ----
    dbh_height: float = 1.3         # m. Altura padrao de medicao
    adaptive_dbh: bool = True       # True: mede acima da raiz mais alta quando ela passa de dbh_height
    stable_window: int = 4          # fatias do topo usadas como referencia de raio estavel
    root_ratio: float = 1.25        # raio > root_ratio x raio de referencia indica zona de raiz
    root_margin: float = 0.10       # m. Folga acima da zona de raiz
    root_min_slices: int = 2        # fatias consecutivas, a partir da base, com raio alargado para valer como raiz
    root_flag_tol: float = 0.05     # m. So marca ROOT_ADJUSTED se a medicao subiu mais que isso acima de dbh_height
    max_dbh_height: float = 2.5     # m. Acima disso a medicao e marcada HIGH_DBH_HEIGHT
    dbh_halfwindow: float = 0.20    # m. Janela em torno de dbh_height quando nao ha raiz (cilindro de varias fatias)
    min_arc_dbh: float = 120.0      # graus. Arco minimo das fatias usadas no DAP (deteccao aceita min_arc_deg, medida exige mais)
    dbh_outlier_frac: float = 0.25  # fatias com raio fora desta fracao da mediana sao descartadas do DAP
    max_dbh_offset: float = 0.40    # m. Distancia maxima da fatia mais proxima quando nao ha fatia na janela
    dbh_window: float = 0.50        # m. Janela acima da altura de medicao para a mediana do raio
    stable_cv: float = 0.15         # coeficiente de variacao maximo do raio na janela (qualidade)

    # ---- 6. Altura total ----
    height_radius: float = 0.60     # m. Raio horizontal em torno do eixo para procurar o topo
    height_bin: float = 0.10        # m
    height_gap: float = 0.75        # m. Lacuna vertical maxima tolerada ao subir pelo fuste
    height_min_points: int = 3      # pontos minimos por bin (altura antiga, por ocupacao)
    track_step: float = 0.25        # m. Degrau do rastreamento do fuste para cima
    track_radius: float = 0.30      # m. Raio em torno do eixo previsto para aceitar pontos do degrau
    track_min_points: int = 6       # pontos minimos por degrau
    track_box: float = 3.0          # m. Caixa horizontal em torno da base para buscar pontos
    height_pct: float = 99.0        # percentil de z dos pontos do ultimo degrau ocupado

    # ---- 7. Qualidade e parcela ----
    min_quality: float = 0.50       # abaixo disso o fuste vai para a lista de revisao
    min_dbh_cm: float = 0.0         # DAP minimo de inclusao na agregacao
    label_radius_factor: float = 2.5  # raio de rotulagem dos pontos = fator x raio do fuste (minimo 0,15 m)

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Parametros desconhecidos no JSON: {sorted(unknown)}")
        return cls(**d)
