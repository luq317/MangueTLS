"""tls_mangue: extracao semiautomatica de parametros estruturais de manguezal em nuvens TLS.

Pipeline C do projeto TLS - Manguezal (Python puro: numpy, scipy, pandas, scikit-learn, laspy).
Modulos:
  config        parametros criticos (todos ajustaveis)
  io_clouds     leitura de LAS/LAZ/XYZ/PTS e escrita de LAS com tree_id
  preprocess    voxel, SOR e ROR
  ground        solo robusto a lacunas de agua e normalizacao de altura
  stems         deteccao de fustes em varias fatias, filtro de raizes-escora, DAP adaptativo, altura
  plot_metrics  agregacao por parcela
  pipeline      execucao ponta a ponta (linha de comando)
  validation    validacao contra o campo (Etapa 4)
  synthetic     nuvem sintetica para testar o codigo
"""
__version__ = "0.2.0"
