import copy
import logging
import os

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from models import vae_loss
from sklearn.metrics.pairwise import cosine_similarity
### loss2
import copy

from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from igraph import *

'''
El archivo trainers.py centraliza todo el código de entrenamiento de modelos en scDEAL:

    Autoencoders (AE, DAE, VAE, CVAE).

    Modelos de predicción.

    Transferencia de conocimiento mediante DaNN.

    Regularización estructural de embeddings para preservar la heterogeneidad celular.

    Funciones de apoyo para clustering de muestras.
'''

#Esta funcion entrena un modelo Autoencoder AE
def train_AE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    #Entrada:
        #net: red neuronal
        #dataloader =Diccionario con el train y valid dataloaders
        #optimizador = Adam ejmplo
        #loss_fuction funcion de perdida
        #n epocas
        #Scheluder para el learning rate
        #load Si queremos cargar un modelo preentrenado
        #Save path ruta para guadar el modelo

    if(load!=False):#Si queremos cargar un modelo preentrenado, load = Trrue
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    #Calculamos el tamano del dataset
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    #guardamos una copia inciial de los pesos por si es mejor
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    #Entrenamiento por epocas y fase
    #Cada epoca tiene un train y un valid
    #definimos si se esta entrenado o evaluando
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            #Iteracion or lotes, se usa solo x porq es un AE(no hay etiquetas
            #Calculamos la perdida entre la reconstruccion y la entrada original
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                #print(x)
                output = net(x)
                # compute loss
                loss = loss_function(output, x)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                #Solo se optimiza en el train
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters
            #print(epoch_loss)
            #Actualizacion del scheleuder,ajustamos el learning rate en funcion de la funcion de perdida
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            #Guardamos el mejor modelo, es decir guardamos los pesos del modelo con menor perdida de vslidacion
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts,guardamos y devolvemos el mejor modelo
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train

#Esta funcion entrena un modelo Denoising AUtoencoder DAE, la diferencia cin el otro esque el DAE,
#introduce ruido en la entrada para forzar al modelo a aprender representaciones mas robustas
def train_DAE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: len(data_loaders[x].dataset) for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):
                #El cambio con el AE esta aqui genera una mascara binaria aleatoria del 20%
                #q se aplica a x para crear Z, donde algunos valores del vector X se ponen a cero
                #esto fuerzza al modelo a aprender de maner mas profunda para sacar ese valor
                z = x
                y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype= bool),] = 0
                x.requires_grad_(True)
                # encode and decode 
                output = net(z)
                # compute loss
                loss = loss_function(output, x)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters

            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train    


#Anadimos gene weight como parametro
def train_DAE_GEN_IMPORTANT_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",gene_weights=None):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)
        if epoch == 0:
            print("Usando pérdida ponderada con gene_weights (top 20% genes importantes)")


        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):
                #El cambio con el AE esta aqui genera una mascara binaria aleatoria del 20%
                #q se aplica a x para crear Z, donde algunos valores del vector X se ponen a cero
                #esto fuerzza al modelo a aprender de maner mas profunda para sacar ese valor
                z = x
                y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype= bool),] = 0
                x.requires_grad_(True)
                # encode and decode 
                output = net(z)
                # compute loss 
                loss = (((output - x) ** 2) * gene_weights).mean()#modificamos la funcion loss
     

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters

            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train    


#Esta funcion entrena un Variational Autoencoder (VAE), a difereencia de AE
#Vae no aprende directamente de una codificacion fija sino una distibcuion (media y 
#varianza) desde la cual muestre para obetener el vector latente





def train_VAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    #Con AE z seria z=[0.5,1.2,-0.3]
    #COn VAE z seria u=[0.5,1.2,-0.3] o=[0.1,0.1,0.1] media y desviacion estandar
    #y luego con eso obteemos z con z=u+ o*e, donde e es ruido aleatorio 
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train


#Esta funcion entrena un Conditional Variational AutoEncoder(CVAE), es decir un VAE
#condicionado por una etiqueta o categoria(por ejemplo , tipo celular o sensibilidad al farmaco)
def train_CVAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    
    #CVAE no solo codifica X, sino que tambien toma como entrada una variable c ( la condicion)
    #la idea es quiero recosntuir X, pero sabiendo que pertenece a la clase c
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, c) in enumerate(data_loaders[phase]):
                #aqui podemos ver el data_loader devuelve dos cosas por batch:
                    #x:los datos de entrada (expresion genica)
                    #c: la condicion asociada (etiqueta , tipo de celula, respuesta)...
                x.requires_grad_(True)
                # encode and decode 
                output = net(x,c)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train


#Esta funcion entrena un modelo de prediccion supervisada clasica(por ejemplo un MLP)
#para predecir una etiqueta y a partir de una entrada x, lo que hace es predecir la sensibilidad a un
#farmaco despues de haber reucido la dimensionalidad con un AE,VAE, o similar
def train_predictor_model(net,data_loaders,optimizer,loss_function,n_epochs,scheduler,load=False,save_path="model.pkl"):

    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf


    #Cada epoca tien un train y un valid 
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:                                          #Configura el modelo en modod entrenamiento o evaluacion segun la fase
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # N iter s calculated
            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            #Iteracion por lotes, cad lote contiene una entrada y una etiqueta real
            for batchidx, (x, y) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                #Calculo del error, la funcion de perdida suele ser BCE o CROSSentropyloss
                output = net(x)
                # compute loss
                loss = loss_function(output, y)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                #Backpropagation y optimizacion, solo se ajustan los pesos en el entrenamiento
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()#Acumulacion de perdida total
            

            epoch_loss = running_loss / n_iters#Calculo de perdida promedio de la epoca
            print(epoch_loss)
            if phase == 'train':#Utilizamps scheduler para actyalizar el leraning rate
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            #Guardamos el mejor modelo
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts, guardamos modelo entrenado
        torch.save(best_model_wts, save_path)
        
    net.load_state_dict(best_model_wts)           
    #En resumen: entrena un modelo predictor supervisado con:
        #entrenamiento por epocas
        #evaluacion por fases
        #perdida monitoreada
        #almacenamiento del mejor modelo
    return net, loss_train

#entrenamos un modelo de transferencia adversarial llamado ADDA, Adversarial Discriminative Domain ADaptation
#ADDA lo que hace es intentar adaptar un modelo entrenado en un dominio fuente (bulk) para que funcione
#bien en un dominio destino sc, sin necesidad de etiquetas en el dominio destino
def train_ADDA_model( #NO SE USA EN ESTE PROYECTO
    source_encoder, target_encoder, discriminator,
    source_loader, target_loader,
    dis_loss, target_loss,
    optimizer, d_optimizer,
    scheduler,d_scheduler,
    n_epochs,device,save_path="saved/models/model.pkl",
    args=None):
    #source_encoder --> codificador entrenado con datos bulk
    #target encoder --> codificador que queremos entrenar 
    #discriminator -->Red que intenta distinguir entre dominios
    #optimizer --> optimiza el target encoder
    #d_optimizeroptimiza el discriminador
    
    #1 calculamos el numero de muestras de cada dominio y se inicializa contadores de perdida
    target_dataset_sizes = {x: target_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    source_dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}

    dataset_sizes = {x: min(target_dataset_sizes[x],source_dataset_sizes[x]) for x in ['train', 'val']}

    loss_train = {}
    loss_d_train = {}

    #Entenamiento por epocas y fases
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #Preparacion por fases, el encoder bulk se congela y el encoder de sc se entrena
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                source_encoder.eval()
                target_encoder.train()  # Set model to training mode
                discriminator.train()  # Set model to training mode

            else:
                source_encoder.eval()
                target_encoder.eval()  # Set model to evaluate mode
                discriminator.eval()  # Set model to training mode

            running_loss = 0.0
            d_running_loss = 0.0

                #losses, d_losses = AverageMeter(), AverageMeter()
            n_iters = min(len(source_loader[phase]), len(target_loader[phase]))
            source_iter, target_iter = iter(source_loader[phase]), iter(target_loader[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for iter_i in range(n_iters):
                #Iteracion por lotes, se sacan los batches de datos fuentes y destino(sin necesidad de etiquetas en el destino)
                source_data, source_target = source_iter.next()
                target_data, target_target = target_iter.next()
                source_data = source_data.to(device)
                target_data = target_data.to(device)
                s_bs = source_data.size(0)
                t_bs = target_data.size(0)

                #PASO 1
                #Entrenamos el discriminador 
                D_input_source = source_encoder.encode(source_data)
                D_input_target = target_encoder.encode(target_data)
                D_target_source = torch.tensor(
                    [0] * s_bs, dtype=torch.long).to(device)
                D_target_target = torch.tensor(
                    [1] * t_bs, dtype=torch.long).to(device)

                # Add adversarial label    
                D_target_adversarial = torch.tensor(
                    [0] * t_bs, dtype=torch.long).to(device)
                
                # train Discriminator
                # Please fix it here to be a classifier
                D_output_source = discriminator(D_input_source)
                D_output_target = discriminator(D_input_target)
                D_output = torch.cat([D_output_source, D_output_target], dim=0)
                D_target = torch.cat([D_target_source, D_target_target], dim=0)
                d_loss = dis_loss(D_output, D_target)
                #El dsicirminador tarta de distinguir entre bulk y sc
                #se entrena con d_optimizer

                d_optimizer.zero_grad()

                if phase == 'train':
                    d_loss.backward()
                    d_optimizer.step()
                
                d_running_loss += d_loss.item()
                #PASO 2
                #Entrenamos el target encoder para enganar al discirminador 
                #intentamos hacer que target_encoder genere embeddings indistinguibles de los de source_encoder
                D_input_target = target_encoder.encode(target_data)
                D_output_target = discriminator(D_input_target)
                loss = dis_loss(D_output_target, D_target_adversarial)

                optimizer.zero_grad()


                if phase == 'train':

                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
            

            epoch_loss = running_loss/n_iters
            d_epoch_loss = d_running_loss/n_iters


            if phase == 'train':
                scheduler.step(epoch_loss)
                d_scheduler.step(d_epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            d_last_lr = d_scheduler.optimizer.param_groups[0]['lr']

            loss_train[epoch,phase] = epoch_loss
            loss_d_train[epoch,phase] = d_epoch_loss

            logging.info('Discriminator {} Loss: {:.8f}. Learning rate = {}'.format(phase, d_epoch_loss,d_last_lr))
            logging.info('Encoder {} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))

            # if phase == 'val' and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts, guadradmos los modelos
        torch.save(discriminator.state_dict(), save_path+"_d.pkl")
        torch.save(target_encoder.state_dict(), save_path+"_te.pkl")

    #net.load_state_dict(best_model_wts)           
    #EL resultado va a ser un target_encoder adaptado al dominio destino, un discriminador entrenado y perdidas loss_trai y loss_d_train
    return discriminator,target_encoder, loss_train, loss_d_train


#Esta funcion entrena un moidelo de transferencia DANN, elobjetivo es aprendecer a predecir
#la respuestan a farmacos en el dominio objetivo sc alineando las representaciones latentes de los dominios fuentes y objetivo mediante
#una perdida MMD 
'''
train_DaNN_model(...)

    Entrena el modelo DaNN (Deep Adaptation Neural Network).

    Objetivo: alinear las representaciones latentes de bulk y single-cell.

    La pérdida total se compone de:

        Pérdida de clasificación (CrossEntropy) sobre datos bulk.

        Pérdida MMD entre representaciones latentes de bulk y single-cell.

    No requiere etiquetas de datos single-cell.

    Es el modelo de transferencia base.
'''
def train_DaNN_model(net,source_loader,target_loader,
                    optimizer,loss_function,n_epochs,scheduler,dist_loss,weight=0.25,GAMMA=1000,epoch_tail=0.90,
                    load=False,save_path="saved/model.pkl",best_model_cache = "drive",top_models=5):

    #Lo que hace DANN:
        #Usa etiquetas y del domino fuente bulk
        #Usa datos sin etiquetas de dominio objetivo sc
        #Ajusta el modelo para que:
                #Prediga correctamente en el dominio fuente
                #Las representaciones z_bulk y Z_sc sean similares usando MMD loss
    #Entradas principales:
        #source_loader--> bulk con etiquetas
        #target_loader --> sc (sin etiquetas)
        #dist_loss --> funcion MMD
        #weight cuanto peso tiene la MMD en la perdida total
    if(load!=False):
        if(os.path.exists(save_path)):
            try:
                net.load_state_dict(torch.load(save_path))           
                return net, 0
            except:
                logging.warning("Failed to load existing file, proceed to the trainning process.")

        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    mmd_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf


    g_tar_outputs = []
    g_src_outputs = []

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mmd = 0.0

            batch_j = 0
            list_src, list_tar = list(enumerate(source_loader[phase])), list(enumerate(target_loader[phase]))
            n_iters = max(len(source_loader[phase]), len(target_loader[phase]))

            for batchidx, (x_src, y_src) in enumerate(source_loader[phase]):
                _, (x_tar, y_tar) = list_tar[batch_j]
                #En cada batch
                x_tar.requires_grad_(True)
                x_src.requires_grad_(True)

                min_size = min(x_src.shape[0],x_tar.shape[0])

                if (x_src.shape[0]!=x_tar.shape[0]):
                    x_src = x_src[:min_size,]
                    y_src = y_src[:min_size,]
                    x_tar = x_tar[:min_size,]
                    y_tar = y_tar[:min_size,]

                #x.requires_grad_(True)
                # encode and decode 
                
                if(net.target_model._get_name()=="CVAEBase"):
                    #Forward del modelo
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar,y_tar)
                else:
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar)
                # compute loss
                #Perida total
                loss_c = loss_function(y_pre, y_src)      
                loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)

                loss = loss_c + weight * loss_mmd


                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    #Backward calcula los gradientes de la funcion perdida respecto a cada uno de los pesos
                    #del modelo:
                                #Recorre la red hacia atras (de salida a entrada)
                                #Calcula como cada peso afecta a la perdida
                                #Guarda esos valores en cada parametro(.grad)

                    loss.backward(retain_graph=True)
                    # update the weights
                    optimizer.step()#esto actualiza los pesos

                # print loss statistics
                running_loss += loss.item()
                running_mmd += loss_mmd.item()

                # Iterate over batch
                batch_j += 1
                if batch_j >= len(list_tar):
                    batch_j = 0


            #MMD mide que tan diferentes son dos distribuciones (bulk vs sc)
            #Se minimiza para alinear los espacios latentes de ambos dominios
            # Average epoch loss
            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd/n_iters

            # Step schedular
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Savle loss
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            mmd_train[epoch,phase] = epoch_mmd

            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            #Guardamos el mejor modelo,solo guarda el modelo si cumple esa condicion
            if (phase == 'val') and (epoch_loss < best_loss) and (epoch >(n_epochs*(1-epoch_tail))) :
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(net.state_dict())
                # Save model if acheive better validation score
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
 
    #     # Select best model wts
    #     torch.save(best_model_wts, save_path)
        
    # net.load_state_dict(best_model_wts)           
        # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)
    #btenemos un  modelo que reuqiere solo etiquetas del domino bulk, permite
    #usar sc como target no etiquetado , es mas estable que ADDA porq no hya adversarios
    return net, [loss_train,mmd_train]


#Esta funcion genera una lista de aristas con pesos para construir un grafo entre los datos,
#usando el algoritmo k-vecinos mas cercanos (knn)
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    #Entrada: featured matrix --> matriz de caracteristicas (representaciones latentes de celulas
    #distance type metrica de distancia(por defecto euclidean
    #k el numero de vecinos mas cercanos a conectar
    #1 calcula la matriz de distancia entre todos los ptos 
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        #print(distMat)
    edgeList=[]
    
    #construye una lista de arista para el grafo
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j],distMat[i,j]))
    
    #Devuelve una lista llamada edgelist con las aristas del grafo, esto se usa en el DANMODEL2
    return edgeList


#Con esta funcion apliamos el algoritmo de clustering Louvain sobre un grafo previamente cosntruido 
#a partir de calculateKNNgraphDistanceMatrix
#Su objetivo es agrupar nodos(celulas) en comunidades (clusters) basandose en la 
#conectividad del grafo
def generateLouvainCluster(edgeList):
   
    """
    Louvain Clustering using igraph
    """
    #Construimos un grafo con Network
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    #onvertimos a matriz de adyacencia
    W = nx.to_scipy_sparse_matrix(Gtmp, format="coo")
    W = W.todense()
    #Crea un grafo de ipraph con pesos 
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    #aplica el algortimo Louvain
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])
    #Esto devuelve listresult: lista donde el indice es el nodo y el valor es el numero de cluster al que pertence
    #size numero total de clusters detectados, se usa en train
    return listResult, size
    

#Esta funcion extiende el modelo DANN anadiendo regularizacion estructural a traves de clustering
#Louvain y cohesion Intra-Cluster
'''
train_DaNN_model2(...) (versión regularizada)

    Igual que train_DaNN_model, pero añade una regularización estructural adicional sobre los datos single-cell:

        Agrupa las muestras scRNA-seq en clusters (Louvain) usando sus embeddings.

        Penaliza la dispersión dentro de cada cluster usando distancia coseno, fomentando embeddings más coherentes.

    Ideal para preservar la heterogeneidad biológica en los datos sc.
   
    Cuando activas la opción --mod new, el script scmodel.py usa train_DaNN_model2(...).
'''
def train_DaNN_model2(net,source_loader,target_loader,
                    optimizer,loss_function,n_epochs,scheduler,dist_loss,weight=0.25,GAMMA=1000,epoch_tail=0.90,
                    load=False,save_path="save/model.pkl",best_model_cache = "drive",top_models=5,k=10,device="cuda"):

    #El objetivo es entrenar un modelo de adapatacion del domino bulk a sc:
        #1.Prediga bien en el domino fuente
        #2.Alinee las representaciones bulk y sc
        #3.Preservar la estructura interna de los datos sc
                #3.1 Agrupa embeddings sc en clusters (louvain)
                #3.2Penaliza la dispersion interna del cluster( usando distancia coseno)
        
    if(load!=False):
        if(os.path.exists(save_path)):
            try:
                net.load_state_dict(torch.load(save_path))           
                return net, 0,0,0
            except:
                logging.warning("Failed to load existing file, proceed to the trainning process.")

        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    mmd_train = {}
    sc_train = {}
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf


    g_tar_outputs = []
    g_src_outputs = []

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mmd = 0.0
            running_sc =0.0
            
            batch_j = 0
            list_src, list_tar = list(enumerate(source_loader[phase])), list(enumerate(target_loader[phase]))
            n_iters = max(len(source_loader[phase]), len(target_loader[phase]))

            for batchidx, (x_src, y_src) in enumerate(source_loader[phase]):
                _, (x_tar, y_tar) = list_tar[batch_j]
                
                x_tar.requires_grad_(True)
                x_src.requires_grad_(True)

                min_size = min(x_src.shape[0],x_tar.shape[0])

                if (x_src.shape[0]!=x_tar.shape[0]):
                    x_src = x_src[:min_size,]
                    y_src = y_src[:min_size,]
                    x_tar = x_tar[:min_size,]
                    y_tar = y_tar[:min_size,]

                #x.requires_grad_(True)
                # encode and decode 
                
                
                
                if(net.target_model._get_name()=="CVAEBase"):
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar,y_tar)
                else:
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar)
                # compute loss
                #Hasta aqui todo igual que DANN
                encoderrep = net.target_model.encoder(x_tar)#contiene los embeddings del domino sc
                #print(x_tar.shape)
                if encoderrep.shape[0]<k:
                    next
                else:    
                    edgeList = calculateKNNgraphDistanceMatrix(encoderrep.cpu().detach().numpy(), distanceType='euclidean', k=10)#Agrupamos las muestras latentes en clusters
                    listResult, size = generateLouvainCluster(edgeList) #list result nos dice a que clase pertenve cada muestra
                    # sc sim loss
                    loss_s = 0
                    for i in range(size):
                        #print(i)
                        #Calculamos la perdida de cohesion intra cluster, toda esta penalizacion se guarda como loss_s
                        s = cosine_similarity(x_tar[np.asarray(listResult) == i,:].cpu().detach().numpy())
                        s = 1-s
                        loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                    #loss_s = torch.tensor(loss_s).cuda()
                    if(device=="cuda"):
                        loss_s = torch.tensor(loss_s).to(device)
                    else:
                        loss_s = torch.tensor(loss_s).cpu()
                    loss_s.requires_grad_(True)
                    loss_c = loss_function(y_pre, y_src)      
                    loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)
                    #print(loss_s,loss_c,loss_mmd)
    
                    loss = loss_c + weight * loss_mmd +loss_s #Perdida total
    
    
                    # zero the parameter (weight) gradients
                    optimizer.zero_grad()
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        # update the weights
                        optimizer.step()
    
                    # print loss statistics
                    running_loss += loss.item()
                    running_mmd += loss_mmd.item()
                    running_sc += loss_s.item()
                    # Iterate over batch
                    batch_j += 1
                    if batch_j >= len(list_tar):
                        batch_j = 0

            # Average epoch loss
            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd/n_iters
            epoch_sc = running_sc/n_iters
            # Step schedular
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Savle loss
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            mmd_train[epoch,phase] = epoch_mmd
            sc_train[epoch,phase] = epoch_sc
            
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if (phase == 'val') and (epoch_loss < best_loss) and (epoch >(n_epochs*(1-epoch_tail))) :
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(net.state_dict())
                # Save model if acheive better validation score
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
 
    #     # Select best model wts
    #     torch.save(best_model_wts, save_path)
        
    # net.load_state_dict(best_model_wts)           
        # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    #Nos va a decolcer el modelo entrenado , y las curvas de perdida de entraniemto por fase , loss train total
    #mmd_train solo MMd y sc_train solo cohesion

    #DANN:
        #Solo se enfoca en alinear los dominios (bulk y sc)
        #Usar MMD  como unica regularizacion entre dominios
    #DANN2:
        #Alinea dominios ( como DaNN)
        #Y ademas agrupa las muestras de sc en clusters
        #Penaliza si dentro de cada cluster los embeddings estan muy dispersos
        #Con esto frozamos am producir represtnaciones mas coheretnes, especialmente 
        #util si las celulas sc tienen subtipos, estados o grupos
        #EMBEDDING es una represtnacion numerica y densa de un dato en un espacio de menor dimension
        #traducir una celula de 20000 genes a uno mas pequeno con 64 valores 
    return net, loss_train,mmd_train,sc_train    