import logging
import re
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from captum.attr import IntegratedGradients
from pandas import read_excel
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_recall_curve, roc_curve
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, Subset
from pybiomart import Server



'''
El archivo utils.py proporciona un conjunto de funciones auxiliares 
esenciales para las diferentes fases del pipeline de scDEAL, incluyendo preprocesamiento de datos, 
visualización, interpretabilidad del modelo y evaluación del rendimiento.

El archivo utils.py actúa como un complemento esencial que conecta el 
aprendizaje automático con la interpretación biológica. Permite preparar y anotar 
adecuadamente los datos, visualizar el proceso de entrenamiento, 
y sobre todo extraer conocimiento biológico útil a partir de las predicciones del modelo 
(identificación de genes críticos, validación con pseudotiempo, entre otros).
'''

#Con esta funcion seleccionamos los genes altamente variables 
def highly_variable_genes(data, 
    layer=None, n_top_genes=None, 
    min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3, 
    span=0.3, n_bins=20, flavor='seurat', subset=False, inplace=True, batch_key=None, PCA_graph=False, PCA_dim = 50, k = 10, n_pcs=40):

    adata = sc.AnnData(data)#lo convertimos en andata

    adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
    adata.obs_names_make_unique()


    if n_top_genes!=None:
        sc.pp.highly_variable_genes(adata,layer=layer,n_top_genes=n_top_genes,
        span=span, n_bins=n_bins, flavor='seurat_v3', subset=subset, inplace=inplace, batch_key=batch_key)

    else: 
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,
        layer=layer,n_top_genes=n_top_genes,
        min_disp=min_disp, max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, 
        span=span, n_bins=n_bins, flavor=flavor, subset=subset, inplace=inplace, batch_key=batch_key)

    if PCA_graph == True:
        sc.tl.pca(adata,n_comps=PCA_dim)
        X_pca = adata.obsm["X_pca"]
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)

        return adata.var.highly_variable,adata,X_pca


    return adata.var.highly_variable,adata

#Con esta funcion guaramod los argumentos pasdado al script 
def save_arguments(args,now):
    args_strings =re.sub("\'|\"|Namespace|\(|\)","",str(args)).split(sep=', ')
    args_dict = dict()
    for item in args_strings:
        items = item.split(sep='=')
        args_dict[items[0]] = items[1]

    args_df = pd.DataFrame(args_dict,index=[now]).T
    args_df.to_csv("save/logs/arguments_" +now + '.csv')

    return args_df

#Con esta funcion visualizamos la distibuvcion del vector etiquetas Y 
#Generamos un histograma de las etiquetas de salida Y , objetivo ver como estan distibuidas esos datos 
def plot_label_hist(Y,save=None):

    # the histogram of the data
    n, bins, patches = plt.hist(Y, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Y values')
    plt.ylabel('Probability')
    plt.title('Histogram of target')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    if save == None:
        plt.show()
    else:
        plt.savefig(save)

# plot no skill and model roc curves
#Dibuja y guarda la curva ROC comparando:
    #Un clasificador aleatorio (naive_probs
    #Mi modelo entrenado
def plot_roc_curve(test_y,naive_probs,model_probs,title="",path="figures/roc_curve.pdf"):

    # plot naive skill roc curve
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    plt.plot(fpr, tpr, linestyle='--', label='Random')
    # plot model roc curve
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.plot(fpr, tpr, marker='.', label='Predition')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 

# plot no skill and model precision-recall curves
#Dibuja la curva de precision vs Recall(PR CUrve)para evaluar el rendimienti del modelo
# #util cuando hay desbalanceo de clases 
def plot_pr_curve(test_y,model_probs,selected_label = 1,title="",path="figures/prc_curve.pdf"):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(test_y[test_y==selected_label]) / len(test_y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Predition')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 

#detecta el dataset usado y llama a la funcion personalizada para prepararlo correctamente 
def specific_process(adata,dataname="",**kargs):
    if dataname =="GSE117872":
        select_origin = kargs['select_origin']
        adata = process_117872(adata,select_origin=select_origin)#Si usamos GSE117872, llama a esa funcion 
    elif dataname == "GSE122843":
        adata = process_122843(adata)
    elif dataname == "GSE110894":
        adata = process_110894(adata)
    elif dataname == "GSE112274":
        adata = process_112274(adata)
    elif dataname == "GSE108383":
        adata = process_108383(adata)
    elif dataname == "GSE140440":
        adata = process_140440(adata)
    elif dataname == "GSE129730":
        adata = process_129730(adata)
    elif dataname == "GSE149383":
        adata = process_149383(adata)
    return adata
    
def process_108383(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2,3,4]:
        svals = [index.split("_")[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals
    df_annotation=pd.DataFrame(annotation_dict,index=obs_names)
    adata.obs=df_annotation
    # adata.obs['name_section_3'].replace("par", "sensitive", inplace=True)
    # adata.obs['name_section_3'].replace("br", "resistant", inplace=True)
    # adata.obs['sensitive']=adata.obs['name_section_3']

    sensitive = [int(row.find("br")==-1) for row in adata.obs.loc[:,"name_section_3"]]
    sens_ = ['Resistant' if (row.find("br")!=-1) else 'Sensitive' for row in adata.obs.loc[:,"name_section_3"]]
    #adata.obs.loc[adata.obs.cluster=="Holiday","cluster"] = "Sensitive"
    adata.obs['sensitive'] = sensitive
    adata.obs['sensitivity'] = sens_

    # Cluster de score
    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)
    return adata

def process_117872(adata,**kargs):

    annotation = pd.read_csv('data/GSE117872/GSE117872_good_Data_cellinfo.txt',sep="\t",index_col="groups")
    for item in annotation.columns:
        #adata.obs[str(item)] = annotation.loc[:,item].convert_dtypes('category').values
        adata.obs[str(item)] = annotation.loc[:,item].astype("category")

    if "select_origin" in kargs:
        origin = kargs['select_origin']
        if origin!="all":
            selected=adata.obs['origin']==origin
            selected=selected.to_numpy('bool')
            adata = adata[selected, :]
    
    sensitive = [int(row.find("Resistant")==-1) for row in adata.obs.loc[:,"cluster"]]
    sens_ = ['Resistant' if (row.find("Resistant")!=-1) else 'Sensitive' for row in adata.obs.loc[:,"cluster"]]
    #adata.obs.loc[adata.obs.cluster=="Holiday","cluster"] = "Sensitive"
    adata.obs['sensitive'] = sensitive
    adata.obs['sensitivity'] = sens_

    # Cluster de score
    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)
    return adata

def process_122843(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE122843/GSE122843_CellInfo.xlsx' # change it to the name of your excel file
    df_cellinfo = read_excel(file_name,header=2)
    df_cellinfo = df_cellinfo.fillna(method='pad')

    # Dictionary of DMSO between cell info and expression matrix
    match_dict={'DMSO':'DMSO (D7)',
            "DMSOw8":'DMSO (D56)',
            "IBET400":"400nM IBET",
           "IBET600":"600nM IBET",
           "IBET800":"800nM IBET",
           "IBETI1000":"1000nM IBET",
           "IBET1000w8":"1000nM IBET (D56)"}
    inv_match_dict = {v: k for k, v in match_dict.items()}

    index = [inv_match_dict[sn]+'_' for sn in df_cellinfo.loc[:,'Sample Name']]

    # Creat index in the count matrix style
    inversindex = index+df_cellinfo.loc[:,'Well Position']
    inversindex.name = 'Index'
    df_cellinfo.index = inversindex

    # Inner join of the obs adata information
    obs_merge = pd.merge(adata.obs,df_cellinfo,left_index=True,right_index=True,how='left')

    # Replace obs
    adata.obs = obs_merge
    
    return adata

#Con estas funciones preparamos el datset:
    #leemos el archivo , unimos el adata , vemos si la celula es senible o resitente , calculamos los genes mas importantes
    #Devuelve: el mismo adata pero mas potente: Tiene metadatos, sabe si la celula es sensible o resitente , tiene scores de genes importsntes
def process_110894(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE110894/GSE110894_CellInfo.xlsx' # change it to the name of your excel file
    df_cellinfo = read_excel(file_name,header=3)
    df_cellinfo=df_cellinfo.dropna(how="all")
    df_cellinfo = df_cellinfo.fillna(method='pad')
    well_post = ["_"+wp.split("=")[0] for wp in df_cellinfo.loc[:,'Well position']]
    inversindex = df_cellinfo.loc[:,'Plate#']+well_post
    inversindex.name = 'Index'
    df_cellinfo.index = inversindex
    obs_merge = pd.merge(adata.obs,df_cellinfo,left_index=True,right_index=True,how='left')
    adata.obs = obs_merge
    sensitive = [int(row.find("RESISTANT")==-1) for row in obs_merge.loc[:,"Sample name"]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("RESISTANT")!=-1) else 'Sensitive' for row in obs_merge.loc[:,"Sample name"]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata


def process_112274(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2,3]:
        svals = [index.split("_")[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals
    df_annotation=pd.DataFrame(annotation_dict,index=obs_names)
    adata.obs=df_annotation

    sensitive = [int(row.find("parental")!=-1) for row in df_annotation.loc[:,"name_section_2"]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("parental")==-1) else 'Sensitive' for row in df_annotation.loc[:,"name_section_2"]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    

    return adata

def process_116237(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2]:
        svals = [re.split('_|\.',index)[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals  

    return adata

def process_140440(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE140440/Annotation.txt' # change it to the name of your excel file
    df_cellinfo = pd.read_csv(file_name,header=None,index_col=0,sep="\t")
    sensitive = [int(row.find("Res")==-1) for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("Res")!=-1) else 'Sensitive' for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata

def process_129730(adata,**kargs):
    #Data specific preprocessing of cell info
    # sensitive = [ 1 if row in [''] \
    #                 for row in adata.obs['sample']]
    sensitive = [ 1 if (row <=9) else 0 for row in adata.obs['sample'].astype(int)]
    adata.obs['sensitive'] = sensitive
    sens_ = ['Resistant' if (row >9) else 'Sensitive' for row in adata.obs['sample'].astype(int)]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata
    
def process_149383(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE149383/erl_total_2K_meta.csv' # change it to the name of your excel file
    df_cellinfo = pd.read_csv(file_name,header=None,index_col=0)
    sensitive = [int(row.find("res")==-1) for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("res")!=-1) else 'Sensitive' for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata


#Sirve para interpretar el modelo y ver que genes son importantes para predecir la sensibilidad 
def integrated_gradient_check(net,input,target,adata,n_genes,target_class=1,test_value="expression",save_name="feature_gradients",batch_size=100):
       #Aplicamos Integrated Gradients de Captum 
        ig = IntegratedGradients(net)
        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()
        #guardamos la media de importacia por gen 
        adata.var['integrated_gradient_sens_class'+str(target_class)] = attr.mean(axis=0)


        #Separamos muestras sensibles y resitentes
        sen_index = (target == 1)
        res_index = (target == 0)

        # Add col names to the DF
        attr = pd.DataFrame(attr, columns = adata.var.index)

        # Construct attr as a dafaframe
        #Obtenemos los genes mas y menos importantes
        df_top_genes = adata.var.nlargest(n_genes,"integrated_gradient_sens_class"+str(target_class),keep='all')
        df_tail_genes = adata.var.nsmallest(n_genes,"integrated_gradient_sens_class"+str(target_class),keep='all')
        list_topg = df_top_genes.index 
        list_tailg = df_tail_genes.index 

        top_pvals = []
        tail_pvals = []

        if(test_value=='gradient'):
            feature_sens = attr[sen_index]
            feature_rest = attr[res_index]
        else:        
            expression_norm = input.detach().cpu().numpy()
            expression_norm = pd.DataFrame(expression_norm, columns = adata.var.index)
            feature_sens = expression_norm[sen_index]
            feature_rest = expression_norm[res_index]

        for g in list_topg:
            f_sens = feature_sens.loc[:,g]
            f_rest = feature_rest.loc[:,g]
            stat,p =  mannwhitneyu(f_sens,f_rest)#compara expresion entre sensibles y resitentes (con test MANN-WHitney U)
            top_pvals.append(p)

        for g in list_tailg:
            f_sens = feature_sens.loc[:,g]
            f_rest = feature_rest.loc[:,g]
            stat,p =  mannwhitneyu(f_sens,f_rest)
            tail_pvals.append(p)

        df_top_genes['pval']=top_pvals
        df_tail_genes['pval']=tail_pvals


        #Guardamos resultados en el CSV
        df_top_genes.to_csv("save/results/top_genes_class" +str(target_class)+ save_name + '.csv')
        df_tail_genes.to_csv("save/results/top_genes_class" +str(target_class)+ save_name + '.csv')

        return adata,attr,df_top_genes,df_tail_genes


#Tambien usa Integrated Gradient para encontrar genes importantes pero lo hace comparando
#globalmente los grupos sensibles y resitentes 
def integrated_gradient_differential(net,input,target,adata,n_genes=None,target_class=1,clip="abs",save_name="feature_gradients",ig_pval=0.05,ig_fc=1,method="wilcoxon",batch_size=100):
        
        # Caculate integrated gradient
        ig = IntegratedGradients(net)

        df_results = {}

        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()

        #Procesamos los gradietnes segun clip, nos quedamos con positivos o negativos 
        if clip == 'positive':
            attr = np.clip(attr,a_min=0,a_max=None)
        elif clip == 'negative':
            attr = abs(np.clip(attr,a_min=None,a_max=0))
        else:
            attr = abs(attr)

        igadata= sc.AnnData(attr)#Creamos un nuevo AnnnDATA 
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index

        igadata.obs['sensitive'] = target
        igadata.obs['sensitive'] = igadata.obs['sensitive'].astype('category')

        #Aplica analisis de expresion difernecial sobre los gradientes , compar las importancias media
        #de cada gen entre los sensibles y resitentes , usa WIlcoxon 
        sc.tl.rank_genes_groups(igadata, 'sensitive', method=method,n_genes=n_genes)

        for label in [0,1]:

            try:
                df = sc.get.rank_genes_groups_df(igadata, group=label)#Filtra genes con criterio estadistico 
                df_degs = df_degs.loc[(df_degs.pvals_adj<ig_pval) & (df_degs.logfoldchanges>=ig_fc)]
               #Guardamos resuktados por clase (0 o 1)
                df_degs.to_csv("save/results/DIG_class_" +str(target_class)+"_"+str(label)+ save_name + '.csv')

                df_results[label]= df_degs
            except:
                logging.warning("Only one class, no two calsses critical genes")

        #Devuelve el adata original sin modificar el igadata (anndata con gradietnes) , lista de genes
        #importantes en resistentes (clase 0) , lista de genes importantes en sensibles (clase 1)
        return adata,igadata,list(df_results[0].names),list(df_results[1].names)

#COn esta funcion encontramos genes difernecialmente expresados entre grupos ( sensi vs resit) 
#y luego calculamos un score para cada celula basado en esos genes, con esos genes
#calculamos un score que te dice si la celula parece mas sensible o resitennte segun esos genes
def de_score(adata,clustername,pval=0.05,n=50,method="wilcoxon",score_prefix=None):
    #Entrada : adata,
            #Clustername: nombre de la co;una adtaa.obs que define los grupos (ejemplo:sensitivity)
            #pval: umbral de valor-p para considerar un gen significativo
            #n : numero de genes a considerar por grupo
            #method: test estadistico a usar ( Wilconxon o t-test)
            #score_prefix: no usado en el codfigo actual
    try:
        sc.tl.rank_genes_groups(adata, clustername, method=method,use_raw=True)
    except:
        sc.tl.rank_genes_groups(adata, clustername, method=method,use_raw=False)
    # Cluster de score
    for cluster in set(adata.obs[clustername]):
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        select_df = df.iloc[:n,:]
        if pval!=None:
            select_df = select_df.loc[df.pvals_adj < pval]
        sc.tl.score_genes(adata, select_df.names,score_name=str(cluster)+"_score" )
    return adata


#Esta funcion grafica la perdida de entrenamiento y validacion durante las epocas del entrenamiento del modelo
#Sirve para: ver como fue el proceso de aprendizaje durante las 100 epocas
    #Ver si el modelo esta aprendiendo correctamente
    #Detectar overfitting (cuando la perdida de validacion sube mientras la de entrenamienti baja)
def plot_loss(report,path="figures/loss.pdf",set_ylim=False):

    train_loss = []
    val_loss = []


    epochs = int(len(report)/2)
    print(epochs)

    score_dict = {'train':train_loss,'val':val_loss}

    for phrase in ['train','val']:
        for i in range(0,epochs):
            score_dict[phrase].append(report[(i,phrase)])
    plt.close()
    plt.clf()
    x = np.linspace(0, epochs, epochs)
    plt.plot(x,val_loss, '-g', label='validation loss')
    plt.plot(x,train_loss,':b', label='trainiing loss')
    plt.legend(["validation loss", "trainiing loss"], loc='upper left')
    if set_ylim!=False:
        plt.ylim(set_ylim)
    plt.savefig(path)
    plt.close()
    #Para interpretarlo:
        #Modelo entrenado bien: AMbas curvas bajan y se estabilizan 
        #Overfitting: entrenamiento baja mucho pero validacion empieza a subir
        #Underfitting o modelo muy flojo: AMbas curvas se mantienen altas y no mejoran
    return score_dict

#CALCULAR IMPORTANCIA DE GENES
def calculate_gene_weights(expression_matrix, top_percentage=0.2):
    """
    Calcula pesos para los genes basados en su varianza.
    
    Args:
        expression_matrix (numpy.ndarray): Matriz de expresión (cells x genes)
        top_percentage (float): Porcentaje de genes más variables que se consideran importantes (ej: 0.2 para top 20%)

    Returns:
        torch.Tensor: Vector de pesos (n_genes,) como tensor de PyTorch
    """
    variances = np.var(expression_matrix, axis=0)  # calcular varianza de cada gen
    n_genes = len(variances)
    n_top = int(top_percentage * n_genes)

    # Índices de los genes más variables
    top_gene_indices = np.argsort(variances)[-n_top:]

    # Crear vector de pesos
    weights = np.ones(n_genes, dtype=np.float32)
    weights[top_gene_indices] = 2.0

    # Convertir a tensor de PyTorch
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights

#SOLO 20% DE LOS GENES 
def select_top_variable_genes(expression_matrix, top_percentage=0.2):
    """
    Filtra la matriz de expresión y devuelve solo los genes más variables (top %).

    Args:
        expression_matrix (numpy.ndarray o pd.DataFrame): Matriz de expresión (muestras x genes)
        top_percentage (float): Porcentaje superior de genes más variables a conservar (ej. 0.2 para el 20%)

    Returns:
        filtered_matrix: Matriz de expresión reducida con solo los genes más variables
        selected_indices: Índices o nombres de los genes seleccionados
    """
    # Paso 1: calcular la varianza de cada gen (por columnas)
    if isinstance(expression_matrix, pd.DataFrame):
        variances = expression_matrix.var(axis=0).values
        gene_names = expression_matrix.columns
    else:
        variances = np.var(expression_matrix, axis=0)
        gene_names = np.arange(expression_matrix.shape[1])

    # Paso 2: seleccionar los índices del top % de genes más variables
    n_genes = len(variances)
    n_top = int(top_percentage * n_genes)
    top_gene_indices = np.argsort(variances)[-n_top:]  # Últimos n_top con mayor varianza

    # Paso 3: filtrar la matriz
    if isinstance(expression_matrix, pd.DataFrame):
        filtered_matrix = expression_matrix.iloc[:, top_gene_indices]
        selected_names = gene_names[top_gene_indices]
    else:
        filtered_matrix = expression_matrix[:, top_gene_indices]
        selected_names = top_gene_indices

    return filtered_matrix, selected_names


from pybiomart import Server

# Función para mapear genes humanos a ratón usando Ensembl
def map_human_to_mouse_genes(human_genes):
    server = Server(host='http://www.ensembl.org')
    human_dataset = server.datasets['hsapiens_gene_ensembl']
    mouse_dataset = server.datasets['mmusculus_gene_ensembl']

    # Consultamos las homologías entre humanos y ratón
    homologs = []
    for gene in human_genes:
        homolog = human_dataset.query(attributes=['ensembl_gene_id', 'mmusculus_homolog_ensembl_gene'])
        homolog = homolog[homolog['ensembl_gene_id'] == gene]
        homologs.append(homolog)

    # Extrayendo los nombres de los genes de ratón correspondientes
    mouse_genes = [homolog['mmusculus_homolog_ensembl_gene'].iloc[0] if len(homolog) > 0 else None for homolog in homologs]

    return mouse_genes



#CALCULAR CURRICULUM LEARNING

def get_curriculum_dataloader(X_tensor, C_tensor, batch_size=128):
    """
    Crea un dataloader agrupando primero las células de clusters más grandes (más fáciles).

    Args:
        X_tensor (torch.Tensor): matriz de expresión (células x genes)
        C_tensor (torch.Tensor): vector de labels de cluster (leiden)
        batch_size (int): tamaño de batch

    Returns:
        dict: {'train': DataLoader} (con currículum aplicado)
    """
    # Agrupar índices por cluster
    cluster_to_indices = defaultdict(list)
    for idx, cluster in enumerate(C_tensor.cpu().numpy()):
        cluster_to_indices[cluster].append(idx)

    # Ordenar clusters de menor a mayor dificultad (por tamaño)
    sorted_clusters = sorted(cluster_to_indices.items(), key=lambda x: len(x[1]),reverse=True)

    # Acumular índices de los clusters (más grandes primero)
    all_indices = []
    for _, indices in sorted_clusters:
        all_indices += indices

    # Crear subset ordenado
    subset = Subset(TensorDataset(X_tensor, C_tensor), all_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return {'train': loader}

def integrated_gradient_differential(net,input,target,adata,n_genes=None,target_class=1,clip="abs",save_name="feature_gradients",ig_pval=0.05,ig_fc=1,method="wilcoxon",batch_size=100):
        
        # Caculate integrated gradient
        ig = IntegratedGradients(net)

        df_results = {}

        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()

        if clip == 'positive':
            attr = np.clip(attr,a_min=0,a_max=None)
        elif clip == 'negative':
            attr = abs(np.clip(attr,a_min=None,a_max=0))
        else:
            attr = abs(attr)

        igadata= sc.AnnData(attr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index

        igadata.obs['sensitive'] = target
        igadata.obs['sensitive'] = igadata.obs['sensitive'].astype('category')

        sc.tl.rank_genes_groups(igadata, 'sensitive', method=method,n_genes=n_genes)

        for label in [0,1]:

            try:
                df = sc.get.rank_genes_groups_df(igadata, group=label)
                df_degs = df_degs.loc[(df_degs.pvals_adj<ig_pval) & (df_degs.logfoldchanges>=ig_fc)]
                df_degs.to_csv("save/results/DIG_class_" +str(target_class)+"_"+str(label)+ save_name + '.csv')

                df_results[label]= df_degs
            except:
                logging.warning("Only one class, no two calsses critical genes")

        return adata,igadata,list(df_results[0].names),list(df_results[1].names)

