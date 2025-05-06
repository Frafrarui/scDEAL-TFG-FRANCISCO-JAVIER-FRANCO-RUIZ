#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
from pandas.core.frame import DataFrame
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
import utils as ut
from captum.attr import IntegratedGradients
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase)
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random
seed = 42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

'''
Su objetivo principal es aplicar aprendizaje profundo por transferencia 
(transfer learning) para predecir la sensibilidad a fármacos en células individuales, 
usando el conocimiento aprendido previamente con datos bulk.

El script scmodel.py implementa los pasos 3, 4 y 5 del pipeline de scDEAL:
    Entrena un encoder para datos single-cell.
    Aplica transferencia de conocimiento desde bulk con un modelo DaNN.
    Realiza predicciones célula a célula e interpreta qué genes fueron más importantes.
'''

DATA_MAP={#Diccinionario par cuando le pasemos un arvhivo sepa cual es
"GSE117872":"data/GSE117872/GSE117872_good_Data_TPM.txt",
"GSE110894":"data/GSE110894/GSE110894.csv",
"GSE112274":"data/GSE112274/GSE112274_cell_gene_FPKM.csv",
"GSE140440":"data/GSE140440/GSE140440.csv",
"GSE149383":"data/GSE149383/erl_total_data_2K.csv",
"GSE110894_small":"data/GSE110894/GSE110894_small.h5ad"
}
class TargetModel(nn.Module):#clase de pytorch q combina 
    def __init__(self, source_predcitor,target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor # aqui recibe un mdelo ya  esta entrenado como Pretrained Predictor
        self.target_encoder = target_encoder #Un autoencoder que sabe convertir datos de sc en representaciones latentes

    def forward(self, X_target,C_target=None):
# recibe           datos de celula unica   etiqueta de cluster solo si usamos CVAE
        if(type(C_target)==type(None)):
            x_tar = self.target_encoder.encode(X_target)#Si no pasamos C_target solo codfica X_target, resultaso una representacion de sc (encoder)
        else:
            x_tar = self.target_encoder.encode(X_target,C_target)
        y_src = self.source_predcitor.predictor(x_tar)#prediccion final
        return y_src
        
def run_main(args): 
################################################# START SECTION OF LOADING PARAMETERS #################################################
    # Read parameters

    t0 = time.time()#Guaramos la hora actual 
# Con esto recoge los argumentos que le hemos pasado desde la consolas
    # Overwrite params if checkpoint is provided
    #args.checkpoint = "save/sc_pre/integrate_data_GSE112274_drug_GEFITINIB_bottle_256_edim_512,256_pdim_256,128_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_no_DaNN.pkl"
    if(args.checkpoint not in ["False","True"]):#En caso de haberle pasado un checkpoint busca un archivo .pkl
        selected_model = args.checkpoint#Extremos los parmetro del checkpoint
        split_name = selected_model.split("/")[-1].split("_")
        para_names = (split_name[1::2])
        paras = (split_name[0::2])
        args.bulk_h_dims = paras[4]
        args.sc_h_dims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]
                # Nuevo: Imprimir si el modelo tiene gene importance
        print(f"[DEBUG] use_curriculum = {args.use_curriculum} (type: {type(args.use_curriculum)})")

        if "GenImpo" in args.dimreduce:
            print("✅ Modelo cargado con importancia de genes (DAE + Gene Importance)")
            args.dimreduce = args.dimreduce.replace("GenImpo", "")  # Quitamos GenImpo para que DAE siga funcionando normal
        else:
            print("✅ Modelo cargado normal (DAE sin Gene Importance)")


                


    #Determinar que archivo usar como input, depende del que le pasemos , si no usar lo qu le pasamos como ruta
        if(paras[0].find("GSE117872")>=0):
            args.sc_data = "GSE117872"
            args.batch_id = paras[1].split("GSE117872")[1]
        elif(paras[0].find("MIX-Seq")>=0):
            args.sc_data = "MIX-Seq"
            args.batch_id = paras[1].split("MIX-Seq")[1]    
        else:
            args.sc_data = paras[1]

    # Laod parameters from args
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    na = args.missing_value
    if args.sc_data=='GSE117872_HN120':
        data_path = DATA_MAP['GSE117872']
    elif args.sc_data=='GSE117872_HN137':
        data_path = DATA_MAP['GSE117872']
    elif args.sc_data in DATA_MAP:
        data_path = DATA_MAP[args.sc_data]
    else:
        data_path = args.sc_data
    test_size = args.test_size
    select_drug = args.drug.upper()
    freeze = args.freeze_pretrain
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.bulk_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    data_name = args.sc_data
    label_path = args.label
    reduce_model = args.dimreduce #tipo de modelo AE,VAE ,DAE
    predict_hdims = args.predictor_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    leiden_res = args.cluster_res
    load_model = bool(args.load_sc_model)
    mod = args.mod
    
    # Merge parameters as string for saving model and logging
    #para--->constuimos el nombre del experimento 
    # Antes de construir 'para', ponemos un sufijo
     # Primero añade una variable que detecte si tienes prioridad
    prioritized_suffix = ""
    if getattr(args, "use_prioritized_loss", False):
        prioritized_suffix += "GenImpo"
    if getattr(args, "use_curriculum", False):
        prioritized_suffix += "Curriculum"


    # Ahora construyes 'para' incluyendo prioritized_suffix
    para = str(args.bulk) + "_data_" + str(args.sc_data) + "_drug_" + str(args.drug) + \
        "_bottle_" + str(args.bottleneck) + "_edim_" + str(args.bulk_h_dims) + \
        "_pdim_" + str(args.predictor_h_dims) + "_model_" + reduce_model + prioritized_suffix + \
        "_dropout_" + str(args.dropout) + "_gene_" + str(args.printgene) + \
        "_lr_" + str(args.lr) + "_mod_" + str(args.mod) + "_sam_" + str(args.sampling)

    source_data_path = args.bulk_data
    
    # Record time
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize logging and std out, log y errores
    out_path = log_path+now+"transfer.err"
    log_path = log_path+now+"transfer.log"
    out=open(out_path,"w")
    sys.stderr=out
    
    #Logging parameters, sirve para el log 
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    logging.info("Start at " + str(t0))

    # Create directories if they do not exist
    for path in [args.logging_file,args.bulk_model_path,args.sc_model_path,args.sc_encoder_path,"save/adata/"]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
    
    # Save arguments
    # Overwrite params if checkpoint is provided
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
        split_name = selected_model.split("/")[-1].split("_")
        para_names = (split_name[1::2])
        paras = (split_name[0::2])
        args.bulk_h_dims = paras[4]
        args.sc_h_dims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]
        
        prioritized_suffix = ""
        if getattr(args, "use_prioritized_loss", False):
            prioritized_suffix += "GenImpo"
        if getattr(args, "use_curriculum", False):
            prioritized_suffix += "Curriculum"

        para = str(args.bulk) + "_data_" + str(paras[1]) + "_drug_" + str(args.drug) + \
            "_bottle_" + str(args.bottleneck) + "_edim_" + str(args.bulk_h_dims) + \
            "_pdim_" + str(args.predictor_h_dims) + "_model_" + args.dimreduce + prioritized_suffix + \
            "_dropout_" + str(args.dropout) + "_gene_" + str(args.printgene) + \
            "_lr_" + str(args.lr) + "_mod_" + str(args.mod) + "_sam_" + str(args.sampling)

        args.checkpoint = "True"



    
    # Añadir sufijo al nombre del encoder si usas curriculum learning
    curriculum_suffix = "_curriculum" if args.use_curriculum else ""
    sc_encoder_path = args.sc_encoder_path + para + curriculum_suffix

    # Las demás rutas se quedan igual
    source_model_path = args.bulk_model_path + para
    target_model_path = args.sc_model_path + para
    print(source_model_path)
    args_df = ut.save_arguments(args,now)#guarda los argumentos usados
################################################# END SECTION OF LOADING PARAMETERS ##############################################################

################################################# START SECTION OF SINGLE CELL DATA REPROCESSING #################################################
    #Cargar y preprocesar los datos de expresion genica de sc
    # Load data and preprocessing
    adata = pp.read_sc_file(data_path)#Cargamos el archivo de expresion genica 
    if data_name == 'GSE117872_HN137':#algunos dataset reuqieren pasos especiales, en este dataset solo filtra las celulas de origen HN137
        adata =  ut.specific_process(adata,dataname='GSE117872',select_origin='HN137')
    elif data_name == 'GSE117872_HN120':    
        adata =  ut.specific_process(adata,dataname='GSE117872',select_origin='HN120')
    elif data_name =='GSE122843':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE110894':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE112274':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE116237':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE108383':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE140440':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE129730':
        adata =  ut.specific_process(adata,dataname=data_name)
    elif data_name =='GSE149383':
        adata =  ut.specific_process(adata,dataname=data_name)
    else:
        adata=adata

    #usamos curriculum
    print(f"[DEBUG] Entramos a verificar curriculum - args.use_curriculum: {args.use_curriculum} (type: {type(args.use_curriculum)})")
    if args.use_curriculum:
        print("✅ Activado Curriculum Learning: calculando vecinos y clusters")
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)


    #PREPROCESADO DATOS SINGLE CELL
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)#eliminamos las celulas con menos de 200 genes expresados 
    sc.pp.filter_genes(adata, min_cells=3)#eliminamos los genes que esten en menos de 3 celulas

    adata = pp.cal_ncount_ngenes(adata)#Calculamos metricasd por celula como:
                                        #Numero de genes expresados por celula
                                        #Numero de moleculas totales

    #Preprocesamientto por filtrado
    if data_name not in ['GSE112274','GSE140440']:
        #Selecciona genes entre min y max, expresado min celulas, minimo de genes por celula, normaliza los datos ,aplica log_transformacion
        adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                            filter_mingenes=args.min_g,normalize=True,log=True)
        
    else:
        #Si es de esos dos datset tamien aplica filtro por porcentaje de genes mitocondriales , util para detectar e baja calidad
        adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,percent_mito = args.percent_mito,
                            filter_mingenes=args.min_g,normalize=True,log=True)

    # Select highly variable genes,seleccionamos los genes mas variables , aquelloc cuya espresion varia mucho, asi 
    #reducimos mas el numero de genes
    sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]#actualixamos adata para que solo contenga esos genes 

    # Preprocess data if spcific process is required
    data=adata.X
    # PCA, aplicamos pca mas clustering para reducirlo
    # Generate neighbor graph
    sc.tl.pca(adata,svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10)
    # Generate cluster labels
    sc.tl.leiden(adata,resolution=leiden_res)
    sc.tl.umap(adata)
    adata.obs['leiden_origin']= adata.obs['leiden']
    adata.obsm['X_umap_origin']= adata.obsm['X_umap']
    data_c = adata.obs['leiden'].astype("long").to_list()

################################################# END SECTION OF SINGLE CELL DATA REPROCESSING ####################################################

################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################
    #Prepare to normailize and split target data, aqui lo q hacemos es convertir los datos en tensores
    
    #Primero aplicamos una normalizacion min max para escalar los valores 
    mmscaler = preprocessing.MinMaxScaler()
    try:
        data = mmscaler.fit_transform(data)
    except:
        logging.warning("Only one class, no ROC")
        # Process sparse data
        data = data.todense()
        data = mmscaler.fit_transform(data)

    # Split data to train and valid set
    # Along with the leiden conditions for CVAE propose
    Xtarget_train, Xtarget_valid, Ctarget_train, Ctarget_valid = train_test_split(data,data_c, test_size=valid_size, random_state=42)

    # Select the device of gpu
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
    # Construct datasets and data loaders
    Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)#Convertimos a tensores
    Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)
    #print(Xtarget_validTensor.shape)
    # Use leiden label if CVAE is applied 
    Ctarget_trainTensor = torch.LongTensor(Ctarget_train).to(device)
    Ctarget_validTensor = torch.LongTensor(Ctarget_valid).to(device)
    #print("C",Ctarget_validTensor )
    X_allTensor = torch.FloatTensor(data).to(device)#Contiene todos los tensores 
    C_allTensor = torch.LongTensor(data_c).to(device)
    
    train_dataset = TensorDataset(Xtarget_trainTensor, Ctarget_trainTensor)#crea el dataset de tensores 
    valid_dataset = TensorDataset(Xtarget_validTensor, Ctarget_validTensor)


    #crea dataloaders, que se encargan de:
        #Dividir los datos en batches
        #barajarlos sin Shuffle = True
        #Entregarlo uno a uno durante el entrenamiento
    Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
    Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    if args.use_curriculum:
        print("✅ Aplicando Curriculum Learning con clusters leiden")
        
        X_tensor = Xtarget_trainTensor
        C_tensor = Ctarget_trainTensor
        print("✅ [DEBUG] Cargando DataLoader con Curriculum Learning...")
        dataloaders_pretrain = ut.get_curriculum_dataloader(X_tensor, C_tensor, batch_size=batch_size)
        dataloaders_pretrain['val'] = Xtarget_validDataLoader


        dataloaders_pretrain['val'] = Xtarget_validDataLoader
    else:
        dataloaders_pretrain = {'train': Xtarget_trainDataLoader, 'val': Xtarget_validDataLoader}

    #Guardamos en un diccionario, lo usaremis mas adelante
    #print('START SECTION OF LOADING SC DATA TO THE TENSORS')
#Al final de esto tenemos:
    #Los datosnormalizados 
    #Divididos en entrenamiento/ validacion
    #Convertidos a tensores 
    #Listos para entrenar el encoder (AE, VAE, DAE)

################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################

################################################# START SECTION OF LOADING BULK DATA  #################################################
    #El objetivo es cargar los datos de bulk, codificar las etiquetas y prepararlas como tensores par entrenar el modelo fuente(Source_model)

    # Read source data
    data_r=pd.read_csv(source_data_path,index_col=0)# matriz de expresion genica
    label_r=pd.read_csv(label_path,index_col=0)#Anotaciones de sensibilidad a farmacos
    if args.bulk == 'old':
        data_r=data_r[0:805]
        label_r=label_r[0:805]
    elif args.bulk == 'new':
        data_r=data_r[805:data_r.shape[0]]
        label_r=label_r[805:label_r.shape[0]]              
    else:
        print("two databases combine")
    label_r=label_r.fillna(na)#Como vimos anteriormente si queremos todo el dataset o solo una parte....

    # Extract labels, filtramos solo las filas donde existen una etiqueta de sensibilidad para el farmaco elegido
    selected_idx = label_r.loc[:,select_drug]!=na
    label = label_r.loc[selected_idx,select_drug]
    data_r = data_r.loc[selected_idx,:]
    label = label.values.reshape(-1,1)

    # Encode labels
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(label)
    dim_model_out = 2 # en caso de que esten escritas como ¨resistant" o ... las pasa a numeros

    # Process source data
    mmscaler = preprocessing.MinMaxScaler()
    source_data = mmscaler.fit_transform(data_r) #escalamos todos los datos

    # Split source data, dividimos en entrenamiento y test
    Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(source_data,label, test_size=test_size, random_state=42)
    Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all,Ysource_train_all, test_size=valid_size, random_state=42)

    # Transform source data
    # Construct datasets and data loaders
    Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)#convertimos a tensores
    Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)

    Ysource_trainTensor = torch.LongTensor(Ysource_train).to(device)
    Ysource_validTensor = torch.LongTensor(Ysource_valid).to(device)

    #creamos los tensor dataset y los dataloaders
    sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
    sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)

    #El dataloader entrega los datos en pequenos batches (pequenos bloques)
    Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
    Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}#lo guardamos en un diccionario
    #print('END SECTION OF LOADING BULK DATA')
################################################# END SECTION OF LOADING BULK DATA  #################################################

################################################# START SECTION OF MODEL CUNSTRUCTION  #################################################
   #Aqui se crean y configuran los modelos de deep learning , tanto el encoder para sc como el modelo ya entrenado en bulk

    # Construct target encoder
    #Segun el tipo de dimreduce  se construye un ecoder o toro 
    if reduce_model == "AE":
        #Los parametros importantes para el encoder son esos
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        loss_function_e = nn.MSELoss()
    elif reduce_model == "VAE":
        encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
    if reduce_model == "DAE":
        encoder = AEBase(input_dim=data.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        if args.use_prioritized_loss:
            print("✅ Usando prioritized loss también en pretraining del encoder SC")
            # loss_function_e será una función que aplica los gene_weights (similar a como hicimos en bulk)
            expression_matrix = Xtarget_train  # Datos single-cell normalizados
            gene_weights = ut.calculate_gene_weights(expression_matrix, top_percentage=0.2)  # Usa tu misma función
            gene_weights = torch.tensor(gene_weights, dtype=torch.float32).to(device)

            def prioritized_loss(output, target):
                return (((output - target) ** 2) * gene_weights).mean()

            loss_function_e = prioritized_loss
        else:
            print("✅ Usando MSELoss normal en pretraining del encoder SC")
            loss_function_e = nn.MSELoss()
            

    logging.info("Target encoder structure is: ")
    logging.info(encoder)
    
    encoder.to(device)
    #Configurar optimizador y funcion de perdida
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    # Binary classification
    dim_model_out = 2 #Significa q queremos una salida con dos clases binarias
    # Load the trained source encoder and predictor
    if reduce_model == "AE":
        #cargamos el modelo preentrenado con datos bulk, que ya fue entrenado usando bulkmodel.py
        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
        
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    if reduce_model == "DAE":
        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
        
        source_model.load_state_dict(torch.load(source_model_path))
        print(" Modelo predictor de bulk cargado correctamente desde:")
        source_encoder = source_model    
    # Load VAE model
    elif reduce_model in ["VAE"]:
        source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
        source_model.load_state_dict(torch.load(source_model_path))

        source_encoder = source_model
    #logging.info("Load pretrained source model from: "+source_model_path)
           
    source_encoder.to(device)#esto nos asegura q lo vamos a ejecutar con lo elegido en --device

    #Despues de esto tenemos dos piezas listas: 
        #encoder: aprende a comprimir la expresion genica usa (single cell)
        #source_encoder: ya sabe predecir la sensibilidad a un farmaco ( Bulk)
        #esto se va a usar en DANN

################################################# END SECTION OF MODEL CUNSTRUCTION  #################################################

################################################# START SECTION OF SC MODEL PRETRAININIG  #################################################
   #Aqui el objetivo es entrenar el encoder con los datos de sc antes de aplicar DANN
    # Pretrain target encoder training
    # Pretain using autoencoder is pretrain is not False
    if(str(args.sc_encoder_path)!='False'): # Comprobamos si se debe hacer preentramiento n
        # Pretrained target encoder if there are not stored files in the harddisk
        
        train_flag = True
        sc_encoder_path = str(sc_encoder_path)
        print("Pretrain=="+sc_encoder_path)
        
        # If pretrain is not False load from check point
        if args.checkpoint != "False":
            if args.use_curriculum:
                print("⚠️ [CURRICULUM] Se fuerza reentrenamiento del encoder aunque haya checkpoint")
                train_flag = True  # ignoramos el checkpoint
            else:
                try:
                    encoder.load_state_dict(torch.load(sc_encoder_path))
                    logging.info("Load pretrained target encoder from " + sc_encoder_path)
                    train_flag = False
                except:
                    logging.info("Loading failed, procceed to re-train model")
                    train_flag = True


        # If pretrain is not False and checkpoint is False, retrain the model
        if args.use_curriculum:
            print("🎯 [CURRICULUM] Pretraining con curriculum activado")
        if train_flag == True:
            #Aqui se llama a la funcion de trainer.py que hace el entrenamiento del encoder
            if reduce_model == "AE":
                encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
            if reduce_model == "DAE":
                encoder,loss_report_en = t.train_DAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
                
                                            
            elif reduce_model == "VAE":
                encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,load=False,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
            #print(loss_report_en)
            logging.info("Pretrained finished")

        # Before Transfer learning, we test the performance of using no transfer performance:
        # Use vae result to predict 
        #una vez entrenado el encoder, comprobamos como funcionaria sin transferencia:
            #1.Codificamos toas las celulas (X_Alltensor) con el encoder
            #2.Esas representaciones se pasan al modelo bulk(source_model)
            #3.El predictor bulk da una probabilidad de sensibilidad para cada celula
        embeddings_pretrain = encoder.encode(X_allTensor)
        print(embeddings_pretrain)
        pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()
        adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:,1] #Guardamos los reusltado
        adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

        # Add embeddings to the adata object
        embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
        adata.obsm["X_pre"] = embeddings_pretrain
        
        print("✅ Pretrain prediction hecha")
        print("✅ pretrain_prob_prediction.shape:", pretrain_prob_prediction.shape)

        print("✅ adata shape:", adata.shape)
        print("✅ adata.obs shape:", adata.obs.shape)
        print("✅ adata.obs.index[:5]:", adata.obs.index[:5])

        print("✅ A punto de guardar pretrain_pred en adata.obs")

        #Con esto lo que conseguimos es ver como seria sin la transferencia de aprendizaje de bulk
        #a single cell , para compararlo despues la probailidad de ser sensible
################################################# END SECTION OF SC MODEL PRETRAININIG  #################################################

################################################# START SECTION OF TRANSFER LEARNING TRAINING #################################################
    #Aqui se entrena un modelo con aprendizaje por transferencia usando DANN, para adaptar
    #lo aprendido en  bulk a sc

    #LO Q HACE ES USAR EL ENCODER ENTRENADO DE SC Y LO AJUSTA PARA Q FUNCIONE CON EL PREDICTOR DE BULK 
    # Using DaNN transfer learning
    # DaNN model
    # Set predictor loss
    print("✅ A punto de empezar entrenamiento DANN...")
    loss_d = nn.CrossEntropyLoss()#definir la perdida para la clasificacion
    #Configurar optimizador y scheleuder como antes vimos 
    optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)
       
    # Set DaNN model, constuimos el modelo DANN
    #DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
    #Como vemos tiene un modelo fuente (ya entrenado con bulk, el predictor) y un modelo objetivo (Encoder de single-cell)
    #DANN Esta definido en mmd.py
    DaNN_model = DaNN(source_model=source_encoder,target_model=encoder,fix_source=bool(args.fix_source))
    DaNN_model.to(device)


    # Set distribution loss 
    #Definir la perdida de dsitibucion MMD (Mide la diferencia entre  las represntaciones de bulk
    # y sc usando MMD, esto perdida es lo que le permite al modleo aprender una represtnacion comun para ambos dominios 

    def loss(x,y,GAMMA=args.mmd_GAMMA):
        result = mmd.mmd_loss(x,y,GAMMA)
        return result

    loss_disrtibution = loss
     
    # Train DaNN model , entrenamos el modelo DANN
    #Se entrenan dos variantes seegun el valor de mod
    logging.info("Trainig using" + mod + "model")
    target_model = TargetModel(source_model,encoder)
    # Switch to use regularized DaNN model or not
    if mod == 'ori':#Entrena sin informacion de los clusters , solo con datos de entrada
        if args.checkpoint == 'True':
            DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                dataloaders_source,dataloaders_pretrain,
                                # Should here be all optimizer d?
                                optimizer_d, loss_d,
                                epochs,exp_lr_scheduler_d,
                                dist_loss=loss_disrtibution,
                                load=target_model_path+"_DaNN.pkl",
                                weight = args.mmd_weight,
                                save_path=target_model_path+"_DaNN.pkl")
        else:
            DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                dataloaders_source,dataloaders_pretrain,
                                # Should here be all optimizer d?
                                optimizer_d, loss_d,
                                epochs,exp_lr_scheduler_d,
                                dist_loss=loss_disrtibution,
                                load=False,
                                weight = args.mmd_weight,
                                save_path=target_model_path+"_DaNN.pkl")
    # Train DaNN model with new loss function                    
    if mod == 'new': #Anade etiqueta a los clusters como informacion adicional 
        #args.checkpoint = 'False'
        if args.checkpoint == 'True':
            DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                            dataloaders_source,dataloaders_pretrain,
                            # Should here be all optimizer d?
                            optimizer_d, loss_d,
                            epochs,exp_lr_scheduler_d,
                            dist_loss=loss_disrtibution,
                            load=selected_model,
                            weight = args.mmd_weight,
                            save_path=target_model_path+"_DaNN.pkl",
                            device=device)#Guarda el modelo DANN entrenado 
        else:
            DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                                dataloaders_source,dataloaders_pretrain,
                                # Should here be all optimizer d?
                                optimizer_d, loss_d,
                                epochs,exp_lr_scheduler_d,
                                dist_loss=loss_disrtibution,
                                load=False,
                                weight = args.mmd_weight,
                                save_path=target_model_path+"_DaNN.pkl",
                                device=device)                        

    encoder = DaNN_model.target_model #actualizamos los modelos encoder y predictor 
    source_model = DaNN_model.source_model
    print("Transfer DaNN finished")
    logging.info("Transfer DaNN finished")
################################################# END SECTION OF TRANSER LEARNING TRAINING #################################################


################################################# START SECTION OF PREPROCESSING FEATURES #################################################
    #El objetivo es usar el encoder + el predivtor entrenado para obteber las predicciones finakes ssobre las celulas individuales y guaradar esas predicciones 
    # Extract feature embeddings 
    # Extract prediction probabilities
    embedding_tensors = encoder.encode(X_allTensor)#tomamos todas las celuas y le aplicamos el encoder, esto represtna la informacion ams importnte de cadaa celula 

    prediction_tensors = source_model.predictor(embedding_tensors)#hacer la prediccion con el predictor entrenado en datos bulk , toma los embedings y da la probabilidad de ser sensible
    embeddings = embedding_tensors.detach().cpu().numpy()#pasamos los embeddings a numpy para trabajar con ellos , esto convierte los tensores en arrays numpy para poder guardarlos y verlos mejor 
    predictions = prediction_tensors.detach().cpu().numpy()
    print("predictions",predictions.shape)
    # Transform predict8ion probabilities to 0-1 labels

    #Guardamos las predicciones en adata.obs
    adata.obs["sens_preds"] = predictions[:,1]
    adata.obs["sens_label"] = predictions.argmax(axis=1)
    adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
    adata.obs["rest_preds"] = predictions[:,0]
    
################################################# END SECTION OF ANALYSIS AND POST PROCESSING #################################################

################################################# START SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################
        # ✅ Borrar claves antiguas antes de guardar
    if "X_pre" in adata.obsm.keys():
        print("✅ Borrando X_pre para evitar error al guardar")
        del adata.obsm["X_pre"]
    if "sens_preds_pret" in adata.obs.columns:
        print("✅ Borrando sens_preds_pret para evitar error al guardar")
        del adata.obs["sens_preds_pret"]
    if "sens_label_pret" in adata.obs.columns:
        print("✅ Borrando sens_label_pret para evitar error al guardar")
        del adata.obs["sens_label_pret"]

    #Esta linea guarda todo el objeto adata en un carchivo .h5ad
    adata.write(f"save/adata/{data_name}_{para}.h5ad")

    print("✅ Archivo grande guardado correctamente")
################################################# END SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################
    #Con esto evaluamos el modelo entrenado y (si se pide) interpretar los genes mas relevantes 
    from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
    report_df = {}
    Y_test = adata.obs['sensitive']#Etiquetas verdaderas
    sens_pb_results = adata.obs['sens_preds']#Probaivlidad de que sea sensible 
    lb_results = adata.obs['sens_label']#Etiquetas predichas
    
    #Y_test ture label
    ap_score = average_precision_score(Y_test, sens_pb_results)#Calculamos las metricas de rendimiento 
    
    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    f1score = report_dict['weighted avg']['f1-score']
    report_df['f1_score'] = f1score
    file = 'save/bulk_f'+data_name+'_f1_score_ori.txt' # Guardamos el resultado en un narcvhivo 
    with open(file, 'a+') as f:
         f.write(para+'\t'+str(f1score)+'\n') 
    print("sc model finished")
    # If print gene is true, then print gene


    print(f"[DEBUG] Valor de args.printgene: {args.printgene}")
    print(f"[DEBUG] Se va a ejecutar análisis de genes importantes") if args.printgene == 'T' else print(f"[DEBUG] NO se ejecuta análisis de genes")

    #print("🧪 Estoy justo antes del if args.printgene")

    if (args.printgene=='T'): #EN caso de activar esa opcion hacemos un analisis para ver lo genes mas importsntes 
    
        print("✅ Se está ejecutando el análisis de genes importantes")

        # Set up the TargetModel
        target_model = TargetModel(source_model,encoder)#Esto combian el encoder con el precitor , modelo usado para predecir 
        sc_X_allTensor=X_allTensor
        
        ytarget_allPred = target_model(sc_X_allTensor).detach().cpu().numpy()#Volvemos a calcular las predicicones 
        ytarget_allPred = ytarget_allPred.argmax(axis=1)
        # Caculate integrated gradient, con esto vemos la importancia de cada gen 
        ig = IntegratedGradients(target_model)
        scattr, delta =  ig.attribute(sc_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        scattr = scattr.detach().cpu().numpy()

        # Save integrated gradient
        igadata= sc.AnnData(scattr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index
        sc_gra = "save/" + data_name +"sc_gradient.txt"
        sc_gen = "save/" + data_name +"sc_gene.csv"
        sc_lab = "save/" + data_name +"sc_label.csv"
        np.savetxt(sc_gra,scattr,delimiter = " ")
        DataFrame(adata.var.index).to_csv(sc_gen)
        DataFrame(adata.obs["sens_label"]).to_csv(sc_lab)
        # Guardar los IG como AnnData (.h5ad) para análisis en scanpy
        igadata.obs["sensitivity"] = adata.obs["sens_label"].astype("category")
        igadata.obs_names = adata.obs.index
        igadata.var_names = adata.var.index

        save_path = os.path.join("results", args.sc_data, args.drug)
        os.makedirs(save_path, exist_ok=True)


        igadata.write(os.path.join(save_path, "attr_integrated_gradient.h5ad"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data 
    parser.add_argument('--bulk_data', type=str, default='data/ALL_expression.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',help='Path of the processed bulk RNA-Seq drug screening annotation')
    parser.add_argument('--sc_data', type=str, default="GSE110894",help='Accession id for testing data, only support pre-built data.')
    parser.add_argument('--drug', type=str, default='I.BET.762',help='Name of the selected drug, should be a column name in the input file of --label')
    parser.add_argument('--missing_value', type=int, default=1,help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    parser.add_argument('--var_genes_disp', type=float, default=0,help='Dispersion of highly variable genes selection when pre-processing the data. \
                         If None, all genes will be selected .default: None')
    parser.add_argument('--min_n_genes', type=int, default=0,help="Minimum number of genes for a cell that have UMI counts >1 for filtering propose, default: 0 ")
    parser.add_argument('--max_n_genes', type=int, default=20000,help="Maximum number of genes for a cell that have UMI counts >1 for filtering propose, default: 20000 ")
    parser.add_argument('--min_g', type=int, default=200,help="Minimum number of genes for a cell >1 for filtering propose, default: 200")
    parser.add_argument('--min_c', type=int, default=3,help="Minimum number of cell that each gene express for filtering propose, default: 3")
    parser.add_argument('--percent_mito', type=int, default=100,help="Percentage of expreesion level of moticondrial genes of a cell for filtering propose, default: 100")

    parser.add_argument('--cluster_res', type=float, default=0.2,help="Resolution of Leiden clustering of scRNA-Seq data, default: 0.3")
    parser.add_argument('--mmd_weight', type=float, default=0.25,help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000,help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")

    # train
    parser.add_argument('--device', type=str, default="cpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_model_path','-s', type=str, default='save/bulk_pre/',help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p',  type=str, default='save/sc_pre/',help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--sc_encoder_path', type=str, default='save/sc_encoder/',help='Path of the pre-trained encoder in the single-cell level')
    parser.add_argument('--checkpoint', type=str, default='True',help='Load weight from checkpoint files, can be True,False, or a file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')

    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=500,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=512,help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="AE",help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int,default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--bulk_h_dims', type=str, default="512,256",help='Shape of the source encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--sc_h_dims', type=str, default="512,256",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="16,8",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--batch_id', type=str, default="HN137",help="Batch id only for testing")
    parser.add_argument('--load_sc_model', type=int, default=0,help='Load a trained model or not. 0: do not load, 1: load. Default: 0')
    
    parser.add_argument('--mod', type=str, default='new',help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--printgene', type= str, default='F', help="Print the critical gene list: T: print.Default: T")
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    # miss
    parser.add_argument('--logging_file', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--sampling', type=str, default='no',help='Samping method of training data for the bulk model traning. \
                        Can be no, upsampling, downsampling, or SMOTE. default: no')
    parser.add_argument('--fix_source', type=int, default=0,help='Fix the bulk level model. Default: 0')
    parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    #
    #Modificaciones 
    parser.add_argument('--use_prioritized_loss', action='store_true', help='Use gene-prioritized reconstruction loss instead of standard MSE loss.')
    parser.add_argument('--use_curriculum', action='store_true', help='Use Curriculum Learning by training first with easy cells.')



    args, unknown = parser.parse_known_args()
    print(f"[DEBUG] Valor de args.printgene: {args.printgene} (type: {type(args.printgene)})")
    run_main(args)
