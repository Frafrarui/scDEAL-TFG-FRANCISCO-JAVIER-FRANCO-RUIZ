import argparse
import logging
import sys
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import  nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

import sampling as sam
import utils as ut
import trainers as t
from models import (AEBase,PretrainedPredictor, PretrainedVAEPredictor, VAEBase)
import matplotlib
import random
seed=42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False



'''
En bulkmodel.py se llevan a cabo los pasos 1 y 2 de la presentacion
Paso 1  Extracción de características bulk

    Se utiliza un DAE (Denoising Autoencoder) para extraer representaciones compactas a partir de los datos bulk.

    Se entrena con función de pérdida MSE (reconstrucción).

    Esto se corresponde con la parte donde se entrena el encoder AEBase o VAEBase y se entrena con:

Paso 2  Predicción de la respuesta a fármacos

    Se conecta el encoder preentrenado con un predictor MLP.

    Se entrena con los datos bulk y sus etiquetas (0 = resistente, 1 = sensible) usando entropía cruzada.

    Esto se realiza en la parte donde se entrena PretrainedPredictor o PretrainedVAEPredictor.
'''

def run_main(args):
    #es la funcion principal

    #args.checkpoint = "save/bulk_pre/integrate_data_GSE112274_drug_GEFITINIB_bottle_256_edim_512,256_pdim_256,128_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_no"
    #Si le hemos pasado un checkpoint (modelo entrenado) el script va a extraer
    #los parametros del archivo y lo usa para configurarlo automaticamente
    #(dimensiones, farmacos ....)
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
        split_name = selected_model.split("/")[-1].split("_")#Divide el nombre del archivo
        para_names = (split_name[1::2])#le damos posiciones a los nombres del archivo
        paras = (split_name[0::2])
        args.encoder_hdims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]

    # Extract parameters, este bloque de codigo esta extrayendo y preparando los
    #argumentos que les pase por el terminal o los de por defecto para usarlos
    #en el modelo
    epochs = args.epochs #accede al valor de epocas para entrenarlo 
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    select_drug = args.drug.upper() #Nombre del farmaco en mayusculas
    na = args.missing_value#que valor usar si faltan datos
    data_path = args.data
    label_path = args.label
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    log_path = args.log
    batch_size = args.batch_size
    encoder_hdims = args.encoder_h_dims.split(",")
    preditor_hdims = args.predictor_h_dims.split(",")
    reduce_model = args.dimreduce
    sampling = args.sampling
    PCA_dim = args.PCA_dim

    encoder_hdims = list(map(int, encoder_hdims) )#Conversion de texto a enteros
    preditor_hdims = list(map(int, preditor_hdims) )#ejemplo ["512","256"] a [512,256]
    load_model = bool(args.load_source_model)#Convierte a booleano

    
    #para--> genera el nombre del archivo que se va a guardar
    # Creamos un sufijo si estamos usando prioritized loss
    suffix = ""
    if args.use_prioritized_loss:
        suffix += "GenImpo"
    if args.use_gene_filter:
        suffix += "TOPGENES"


    #para--> genera el nombre del archivo que se va a guardar
    para = str(args.bulk)+"_data_"+str(args.data_name)+"_drug_"+str(args.drug)+ \
        "_bottle_"+str(args.bottleneck)+"_edim_"+str(args.encoder_h_dims)+ \
        "_pdim_"+str(args.predictor_h_dims)+"_model_"+reduce_model+suffix+ \
        "_dropout_"+str(args.dropout)+"_gene_"+str(args.printgene)+ \
        "_lr_"+str(args.lr)+"_mod_"+str(args.mod)+"_sam_"+str(args.sampling)

    #now ---> guarda la fecha y hora actual
    now=time.strftime("%Y-%m-%d-%H-%M-%S")


    #Nos aseguramos de que todas las carpetas necesarias esten creadas antes de guardar nada 
    for path in [args.log,args.bulk_model,args.bulk_encoder,'save/ori_result','save/figures']:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)#las crea con os.makedirs
            print("The new directory is created!")
    
    #print(preditor_path )
    #model_path = args.bulk_model + para 

    # Load model from checkpoint, con esto lo que hacemos es si el usuario ha pasado
    #un nombre de archivo como --checkpoint( es decir no ha puesto ni True ni False) entonces
    #quiero extraer de ese nombre el identificador del modelo
    if(args.checkpoint not in ["False","True"]):
        para = os.path.basename(selected_model).split("_DaNN.pkl")[0]# con esto sacamos el nombre base del archivo
        args.checkpoint = 'True'#cambia el valor de checkpoint para que sepa que tiene q cargar un modelo ya entrenado

    #Crea la ruta donde se guarda/carga el archivo
    preditor_path = args.bulk_model + para 
    bulk_encoder = args.bulk_encoder+para
   
    # Read data, leemos los archivos .csv con pandas para cargar:
    data_r=pd.read_csv(data_path,index_col=0)#la matriz de expresion genica
    label_r=pd.read_csv(label_path,index_col=0)#las etiquetas
    if args.bulk == 'old':#esto sive para ver que parte de los datos quiero usar 
        data_r=data_r[0:805]#por ejemplo la antigua de la 0 a la 805, imagina que es de GDSSC
        label_r=label_r[0:805]
    elif args.bulk == 'new':#y aqui usamos los nuevos de 805 hacia adelante 
        data_r=data_r[805:data_r.shape[0]]
        label_r=label_r[805:label_r.shape[0]]        
    else:
        print("two databases combine")#Con esto usamos todas las muestras
    label_r=label_r.fillna(na)#rellena valores faltantes
    ut.save_arguments(args,now)#guarda todos los argumentos en un archivo, para que sepamos con que parametros entreneamos el modelo 


    # Initialize logging and std out, con esto creamos dos rutas de archivo:
    out_path = log_path+now+"bulk.err"#guardamos los errores que puedan salir
    log_path = log_path+now+"bulk.log"#guardamos los mensajes de informacion y seguimiento

    out=open(out_path,"w") #rediriguimos los errores que saldrian por pantalla al archivp
    sys.stderr=out
    
    #Congigurar el log principal 
    logging.basicConfig(level=logging.INFO,#si es .info , guarda mensajes importantes
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True #descativmaos los logs de matplotlib no tienen sentido

    logging.info(args)

    # Filter out na values, con esto filtramos las filas que tienen valor distinto de na 
    #Imaginemos que select_drug="GEFITINIB" entonces la salida es 
    #[TRUE, FALSE .....
    selected_idx = label_r.loc[:,select_drug]!=na

    #si hemos pasado un valor con --var_genes_disp, entonces el script llama a esta funcion:
       #1. Calcula la dispersion de cada gen
       #2. Se queda con los mas variables 
       #3. DEvuelve una lista de genes hvg ("highly variable genes 
    if(g_disperson!=None):
        hvg,adata = ut.highly_variable_genes(data_r,min_disp=g_disperson)
        # Rename columns if duplication exist
        data_r.columns = adata.var_names
        # Extract hvgs
        data = data_r.loc[selected_idx,hvg]
    else:
        data = data_r.loc[selected_idx,:] # en cado se no haber seleccionadp la variable  --var_genes_disp


    # Do PCA if PCA_dim!=0, en caso de que hayamos dicho que queremos aplicar el PCA
    if PCA_dim !=0 :
        data = PCA(n_components = PCA_dim).fit_transform(data)
    else:
        data = data
        
    # Extract labels
    label = label_r.loc[selected_idx,select_drug]#extraemos las etiquetas (0,1) para el farmaco seleccionado
    data_r = data_r.loc[selected_idx,:]#guardamos la tabla original no vaya a ser que sea necesaria en un futuro

    #  Si el usuario quiere usar solo los genes más variables
    if args.use_gene_filter:
        print("✅ Filtrando solo los genes más variables (top {:.0f}%)".format(args.top_gene_percentage * 100))

        data_df = pd.DataFrame(data, columns=data_r.columns)  # le pasamos los nombres reales de los genes
        data_df_filtered, selected_genes = ut.select_top_variable_genes(data_df, top_percentage=args.top_gene_percentage)
        data = data_df_filtered.values  # convertimos a array para que siga funcionando igual

        # Guardar los genes seleccionados en un archivo de texto
        with open(f"save/adata/selected_genes_{args.data_name}.txt", "w") as f:
            for gene in selected_genes:
                f.write(str(gene) + "\n")
    # Scaling data,escalamos todos los valores entre 0 y 1 , ya que puede haber genes 
    #con vaklores entre 0 y 1000 y otro entre 0 y 1
    mmscaler = preprocessing.MinMaxScaler()

    data = mmscaler.fit_transform(data)

    label = label.values.reshape(-1,1)#con reshape nos aseguramos que tiene la forma correcta


    le = LabelEncoder()
    label = le.fit_transform(label)
    dim_model_out = 2

    #label = label.values.reshape(-1,1)

    logging.info(np.std(data))
    logging.info(np.mean(data))

    # Split traning valid test set, dividimos los datos en cjt de entrenamiento, validacion y test
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)
    # sampling method, parametro q le pasamos en el terminal , todo depende de ese valor 
    if sampling == "no":
        X_train,Y_train=sam.nosampling(X_train,Y_train)
        logging.info("nosampling")
    elif sampling =="upsampling":
        X_train,Y_train=sam.upsampling(X_train,Y_train)
        logging.info("upsampling")
    elif sampling =="downsampling":
        X_train,Y_train=sam.downsampling(X_train,Y_train)
        logging.info("downsampling")
    elif  sampling=="SMOTE":
        X_train,Y_train=sam.SMOTEsampling(X_train,Y_train)
        logging.info("SMOTE")
    else:
        logging.info("not a legal sampling method")

    # Select the Training device, seleccionamos si queremos GPU o CPU
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

    else:
        device = 'cpu'
    #print(device)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    #logging.info(device)
    print(device)

    # Construct datasets and data loaders, las redes neuronales no pueden trabajar directament con
    #datasets , necesitan un formato llamado tensor, que Pytorch entiende 
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)

    Y_trainTensor = torch.LongTensor(Y_train).to(device)
    Y_validTensor = torch.LongTensor(Y_valid).to(device)

    # Preprocess data to tensor
    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)#entrena el encoder
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    # construct TensorDataset
    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)#entrena el predictor
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)#entrega los datos por bloque
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)
    bulk_X_allTensor = torch.FloatTensor(data).to(device)
    bulk_Y_allTensor = torch.LongTensor(label).to(device)
    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}
    print("bulk_X_allRensor",bulk_X_allTensor.shape)

    #Tenemos distintos tipos de encoder:
            #AE--> autoencoder normal
            #DAE --> autoencoder con ruido
            #VAE --> autoencoder probabilistico
        #Con este codigo creamos el encoder que le hemos pasado como parametro
    
    '''
    Aunque 'AE' y 'DAE' comparten la misma arquitectura (AEBase), se comportan diferente en la práctica porque 
    se entrenan con funciones distintas (train_AE_model vs train_DAE_model).
    '''
    if(str(args.pretrain)!="False"):
        dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model == 'AE':
            encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model =='DAE':            
            encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        
        #if torch.cuda.is_available():
        #    encoder.cuda()

        #logging.info(encoder)
        encoder.to(device)
        #print(encoder)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        # Load from checkpoint if checkpoint path is provided
        if(args.checkpoint != "False"):
            load = bulk_encoder
        else:
            load = False

        '''
        Aqui podemos ver lo mencionado sobre que se entrenan de manera distinta 
        '''
        if reduce_model == "AE":
            encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,load=load,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
        elif reduce_model == "VAE":
            encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                            optimizer=optimizer_e,load=False,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder)
        if reduce_model == "DAE":
            if args.use_prioritized_loss:
                print("Training DAE with prioritized gene loss...")

                # Calcular pesos de los genes
                expression_matrix = X_train  # matriz de expresión escalada
                gene_weights = ut.calculate_gene_weights(expression_matrix, top_percentage=0.2)  # 20% top genes

                encoder, loss_report_en = t.train_DAE_GEN_IMPORTANT_model(
                    net=encoder,
                    data_loaders=dataloaders_pretrain,
                    optimizer=optimizer_e,
                    loss_function=loss_function_e,
                    n_epochs=epochs,
                    scheduler=exp_lr_scheduler_e,
                    load=load,
                    save_path=bulk_encoder,
                    gene_weights=gene_weights.to(device)  # NUEVO argumento
                )
            else:
                print("Training DAE with normal loss...")

                encoder, loss_report_en = t.train_DAE_model(
                    net=encoder,
                    data_loaders=dataloaders_pretrain,
                    optimizer=optimizer_e,
                    loss_function=loss_function_e,
                    n_epochs=epochs,
                    scheduler=exp_lr_scheduler_e,
                    load=load,
                    save_path=bulk_encoder
                )
                
        
        #logging.info("Pretrained finished")

    # Defined the model of predictor, juntamos el encoder con el predictor, ya que con el encoder
    #lo que hacemos es crear muestras y con eso le es mas facil al predictor de predecir la respuesta, ya que loe pasamos solo lo wsencial 

    if reduce_model == "AE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)
    if reduce_model == "DAE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)                                
    elif reduce_model == "VAE":
        model = PretrainedVAEPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                        hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                        pretrained_weights=bulk_encoder,freezed=bool(args.freeze_pretrain),z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
    #print("@@@@@@@@@@@")
    logging.info("Current model is:")
    logging.info(model)
    #if torch.cuda.is_available():
    #    model.cuda()
    model.to(device)

    # Define optimizer, preparamos todo para entrenar el predictor 
    optimizer = optim.Adam(model.parameters(), lr=1e-2)#motor de aprendizaje , le dice al 
                                                       #al modelo como ajustar sus pesos despues de ver los errores
                                                       #usamos ADAM 


    loss_function = nn.CrossEntropyLoss()#le dice al modelo cuanto se esta equivocando

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)#con esto si el modelo no mejora durante un tiempo
                                                                # reduce la tasa de aprendizaje automaticamente 

    # Train prediction model if load is not false, en caso de haber cargado un checkpoint para q continue entranado desde ahi
    #print("1111")
    if(args.checkpoint != "False"):
        load = True
    else:
        load = False
    #Esta es la llamada mas importante , ya que el predictor se entrena con los datos reales 
            #modelo completo (model)
            #los datos ( dataloaders_train)
            #el optimizador , funcion de perdida , numero de epocas
            #si hay checkpoint o no
            #Donde se guardara el modelo
    model,report = t.train_predictor_model(model,dataloaders_train,
                                            optimizer,loss_function,epochs,exp_lr_scheduler,load=load,save_path=preditor_path)
   

   #En caso de que le pidamos los genes importantes 
    if (args.printgene=='T'):
        import scanpypip.preprocessing as pp
        bulk_adata = pp.read_sc_file(data_path)
        #print('pp')
        ## bulk test predict critical gene
        import scanpy as sc
        #import scanpypip.utils as uti
        from captum.attr import IntegratedGradients
        #bulk_adata = bulk_adata
        #print(bulk_adata) 
        bulk_pre = model(bulk_X_allTensor).detach().cpu().numpy()  
        bulk_pre = bulk_pre.argmax(axis=1)
        #print(model)
        #print(bulk_pre.shape)
        # Caculate integrated gradient
        ig = IntegratedGradients(model)#Calculamos cuanto aporta cad gen para que el modelo predigaclase 1(sensible)
        
        df_results_p = {}
        target=1
        #attr es una matriz donde cada valor indica la influencia de cada gen en la decision del modelo
        attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        #delta es un valor indica si la explicacion es estable , no se usa aqui
        #attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()#guardamos los valores de importancia en un .txt, uno por gen, uno por muestra 
        
        np.savetxt("save/"+args.data_name+"bulk_gradient.txt",attr,delimiter = " ")#se guarda un archivo .txt con la importancia de cada gen
        from pandas.core.frame import DataFrame
        DataFrame(bulk_pre).to_csv("save/"+args.data_name+"bulk_lab.csv")#prediccion de clase por muestra 
    dl_result = model(X_testTensor).detach().cpu().numpy()


    lb_results = np.argmax(dl_result,axis=1)#te da la clase predicha (0 o 1) para cad muestra, si da [0.3,0.7] es 
                                            #30%clase 0 , 70%clase 1 por loq  es 1
    #pb_results = np.max(dl_result,axis=1)
    pb_results = dl_result[:,1] # extrae las probailidades deu q sea sensible(1) para cada muestra 

    #con ese valor de arriba vemos precision , recall , f1-score para cada clase
    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    report_df = pd.DataFrame(report_dict).T#traspuesta, ponemos las metricas como filas 
    ap_score = average_precision_score(Y_test, pb_results)#Precision promedio valor ideal cerca
    auroc_score = roc_auc_score(Y_test, pb_results)#Area bajo la curva Roc valor idela cerca de 1

    report_df['auroc_score'] = auroc_score
    report_df['ap_score'] = ap_score
    #Guardamos el reporte 
    report_df.to_csv("save/logs/" + reduce_model + select_drug+now + '_report.csv')

    #logging.info(classification_report(Y_test, lb_results))
    #logging.info(average_precision_score(Y_test, pb_results))
    #logging.info(roc_auc_score(Y_test, pb_results))

    #Entrenamos un modelo muy simple que no aprende nada real, solo predice de forma aleatoria, pero respetando la proporcion 
    #de clases en los datos 
    model = DummyClassifier(strategy='stratified')#si en el entrenamiento habia 70% y 30% resitente , DUmmy mantien esos porcentajes pero asigna de manera aleatoria 
    model.fit(X_train, Y_train) 
    yhat = model.predict_proba(X_test)
    naive_probs = yhat[:, 1]

    #Esto sirve para ver si mi modelo realmente aprendio algo o si fue al alzar,
        #Si mi modelo tiene AUROC= 0,85 y el DUMMy AUROC = 0.52 mi modelo si aprendio

        
    # ut.plot_roc_curve(Y_test, naive_probs, pb_results, title=str(roc_auc_score(Y_test, pb_results)),
    #                     path="save/figures/" + reduce_model + select_drug+now + '_roc.pdf')
    # ut.plot_pr_curve(Y_test,pb_results,  title=average_precision_score(Y_test, pb_results),
    #                 path="save/figures/" + reduce_model + select_drug+now + '_prc.pdf')
    print("bulk_model finished")

if __name__ == '__main__':
#FUNCION PRINCIPAL

    parser = argparse.ArgumentParser()
    #Con argaparse conseguimos que el scriptm pueda recibir parametros
    #desde la linea de comandos 

    # data le damos la ruta a arhivo bulk 
    parser.add_argument('--data', type=str, default='data/ALL_expression.csv',help='Path of the bulk RNA-Seq expression profile')
    #Ruta al arvchivo etiquetas (0=resistente , 1=Sensible)
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',help='Path of the processed bulk RNA-Seq drug screening annotation')
    #Ruta donde se guardara el resumen del entranmiento
    parser.add_argument('--result', type=str, default='save/results/result_',help='Path of the training result report files')
    #Nombre del farmaco a predecir
    parser.add_argument('--drug', type=str, default='I-BET-762',help='Name of the selected drug, should be a column name in the input file of --label')
    #Que valor poner si hay valores faltantes , por defecto esta 1
    parser.add_argument('--missing_value', type=int, default=1,help='The value filled in the missing entry in the drug screening annotation, default: 1')
    #Cuanto del dataset se reserva para test y train
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    #Seleccion de genes mas variables
    parser.add_argument('--var_genes_disp', type=float, default=None,help='Dispersion of highly variable genes selection when pre-processing the data. \
                         If None, all genes will be selected .default: None')
    #Como balancear los datos: no, upsampling , SMOTE etc
    parser.add_argument('--sampling', type=str, default='no',help='Samping method of training data for the bulk model traning. \
                        Can be upsampling, downsampling, or SMOTE. default: no')
    #Numero de componentes de PCA en caso de usarlo
    parser.add_argument('--PCA_dim', type=int, default=0,help='Number of components of PCA  reduction before training. If 0, no PCA will be performed. Default: 0')

    # trainv (Argumentos de entrenamiento)
    #GPU o CPU para entrenar 
    parser.add_argument('--device', type=str, default="cpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    #Carpeta donde guardar (o cargar el encoder
    parser.add_argument('--bulk_encoder','-e', type=str, default='save/bulk_encoder/',help='Path of the pre-trained encoder in the bulk level')
    #Si se hace preentrenamiento (True or False)
    parser.add_argument('--pretrain', type=str, default="True",help='Whether to perform pre-training of the encoder,str. False: do not pretraing, True: pretrain. Default: True')
    #Tasa de aprendizaje
    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    #Numero de epocas de entrenamiento
    parser.add_argument('--epochs', type=int, default=500,help='Number of epoches training. Default: 500')
    #Tamano del batch(cuantas muestras se usan por paso)
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    #Tamano de la capa oculta central ( el resumen de la muestra)
    parser.add_argument('--bottleneck', type=int, default=32,help='Size of the bottleneck layer of the model. Default: 32')
    #Tipo de encoder: AE(autoencoder) o VAE(variacional)
    parser.add_argument('--dimreduce', type=str, default="AE",help='Encoder model type. Can be AE or VAE. Default: AE')
    #Si se congelan los pesos preentrenados (0 no , 1 si)
    parser.add_argument('--freeze_pretrain', type=int, default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    #Tamano de las capas del encoder (como "512,256")
    parser.add_argument('--encoder_h_dims', type=str, default="512,256",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    #Tamano capas del predictor
    parser.add_argument('--predictor_h_dims', type=str, default="16,8",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    #Parametro especial del VAE en caso de usar (encoder)
    parser.add_argument('--VAErepram', type=int, default=1)
    #Nombre corto del dataset predefinido
    parser.add_argument('--data_name', type=str, default="GSE110894",help='Accession id for testing data, only support pre-built data.')
    #Si cargamos un modelo de entrenamiento ejecutado anteriormente 
    parser.add_argument('--checkpoint', type=str, default='True',help='Load weight from checkpoint files, can be True,False, or file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')

    # misc (argumentos opcionales)
    #carpeta donde se guarda el modelo predictor
    parser.add_argument('--bulk_model', '-p',  type=str, default='save/bulk_pre/',help='Path of the trained prediction model in the bulk level')
    #carpeta donde se guarda los logs de entrenamiento
    parser.add_argument('--log', '-l',  type=str, default='save/logs/',help='Path of training log')
    #Quieres cargar un modelo ya entrenado? (1 si , 0 no)
    parser.add_argument('--load_source_model',  type=int, default=0,help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    #new:usa etiquetas de tipo celular; Ori: no las usa
    parser.add_argument('--mod', type=str, default='new',help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    #Se imprimen los genes mas importantes (True or False)
    parser.add_argument('--printgene', type=str, default='F',help='Print the cirtical gene list: T: print. Default: T')
    #Porcentaje de neuronas que se apagan para evitar el sobreajuste
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    #QUe base de datos usar : GDSC, CCLE o integrate ambas
    parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    
    #MODIFICACIONES MIAS
    # Añadir opción para usar pérdida ponderada por genes importantes
    parser.add_argument('--use_prioritized_loss', action='store_true', help='Use gene-prioritized reconstruction loss instead of standard MSE loss.')
    
    # Añadir opción para filtrar solo los genes más variables
    parser.add_argument('--use_gene_filter', action='store_true', help='Filtrar solo el top % de genes más variables antes de entrenar el modelo')
    parser.add_argument('--top_gene_percentage', type=float, default=0.2, help='Porcentaje de genes más variables a conservar (ej. 0.2 = 20%)')


    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    run_main(args)

