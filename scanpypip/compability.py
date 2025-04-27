import pandas as pd
import scanpy as sc
import torch


class AnnTorchDataset(torch.utils.data.Dataset):
    """
    pytorch wrapper for AnnData Datasets
    """

    def __init__(self, adata):

        self.adata = adata

    def __getitem__(self, index):
        return self.adata.X[index]

    def __len__(self):
        return self.adata.shape[0]
    
# Con este script lo que comseguimos es conectar los datos de la libreria scanpy que 
# son de tipo adata al mundo de deeplearning , con esta clase conseguimos tres cosas
# 1.__getitem__(self, index): Acceso a los datos de una celula individual
# 2.__len__(self): saber cuantas celulas hay(numero de filas)
# 3.Compatibilidad de los datos con una red neuronal entrenable, como ya dijimos antes 
