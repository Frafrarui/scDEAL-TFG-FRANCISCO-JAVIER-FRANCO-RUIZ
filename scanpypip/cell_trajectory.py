import scanpy as sc
import preprocessing as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
'''
Qué es trajetory_paga?

Es una función que reconstruye la trayectoria celular (cómo se relacionan 
diferentes tipos de células) usando PAGA 
(Partition-based graph abstraction), una herramienta de Scanpy.
Te devuelve el objeto adata con la información de PAGA añadida y muestra visualizaciones como:

    Un grafo abstracto (PAGA) de cómo se conectan los grupos de células.

    Un mapa de distribución de células coloreado por cluster.

Te ayuda a visualizar la progresión o diferenciación celular y a entender 
cómo se relacionan las poblaciones celulares en un experimento de scRNA-seq.
'''
def trajetory_paga(adata,clustering = "leiden",n_neighbors = 20):
    #Cada nodo representa un cluster de celulas
    #las conexiones muestran como estan relacionada las poblaciones 

    # PCA , primero aplicamos un pca para reducir la dimensionalidad ya que tenemos 
    #miles de genes 
    sc.tl.pca(adata, svd_solver='arpack')

    # Agrupamos las celulas en clusters usando el metodo leiden
    if clustering == "leiden":
        sc.tl.leiden(adata,resolution=0.2)

    # Busca celulas vecinas entre grupos para ver q celulas estan conectadas 
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # Initialize run of paga
    #crea y dibuja un grafo donde cada nodo es un grupo de celulas y cad linea es una
    #conexion entre grupos 
    sc.tl.paga(adata, groups= clustering)
    sc.pl.paga(adata, threshold=0.03)

    # Dibuja como estan distribuidas las celulas en el mapa PAGA con colores por grupo
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color=[clustering], legend_loc='on data')

    return adata
