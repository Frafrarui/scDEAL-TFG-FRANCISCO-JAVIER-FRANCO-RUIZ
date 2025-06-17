# scDEAL documentation
Deep Transfer Learning of Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data

## EJECUCION DEL MODELO
1. Primero tenemos que situarnos en el directorio del trabajo cd /home/franciscojavier/Escritorio/TFG/scDEAL
2. Activar el entorno —> source scDEALenv/bin/activate
3. Ejecutar el modelo desde cero que primero empezara con los datos bulk:
python bulkmodel.py \
--data data/ALL_expression.csv \
--label data/ALL_label_binary_wf.csv \
--drug I.BET.762 \
--data_name GSE110894 \
--dimreduce DAE \
--pretrain True \
--bottleneck 512 \
--encoder_h_dims 256,128 \
--predictor_h_dims 128,64 \
--PCA_dim 0 \
--sampling upsampling \
--epochs 500 \
--batch_size 200 \
--dropout 0.1 \
--bulk integrate \
--mod new \
--printgene F \
--checkpoint False \
--lr 0.5 \
--device cpu

4. Ahora ejecutamos el scmodel.py:
python scmodel.py \
--sc_data "GSE110894" \
--dimreduce "DAE" \
--drug "I.BET.762" \
--bulk_h_dims "256,128" \
--bottleneck 512 \
--predictor_h_dims "128,64" \
--dropout 0.1 \
--printgene "F" \ #Hemos dicho q no nos imprima los genes mas importante
--mod "new" \
--lr 0.5 \
--sampling "upsampling" \
--checkpoint "save/bulk_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_T_lr_0.5_mod_new_sam_upsampling"
Cargó el checkpoint del modelo de predicción en datos bulk (el predictor).

Las predicciones estan guardadas en save/adata

## Instalación de scDEAL

Se recomienda instalar scDEAL en un sistema Linux y configurar el entorno de conda proporcionado mediante conda-pack.

Puedes descargar el paquete comprimido scdeal.tar.gz desde los siguientes enlaces:

    Descargar desde OneDrive

    Descargar desde Google Drive

Se recomienda instalar este entorno dentro del entorno raíz de conda, de forma que el comando conda-pack esté disponible también en cualquier subentorno.

### Instalación de conda-pack
Opción 1: mediante conda (Anaconda o conda-forge)

conda install conda-pack
conda install -c conda-forge conda-pack

Opción 2: mediante PyPI (requiere tener conda instalado previamente)

pip install conda-pack

Cargar el entorno scDEALenv

El comando conda-pack permite desempaquetar el entorno descargado y activarlo. Un caso de uso habitual es preparar el entorno en una máquina para utilizarlo en otra que no tenga conda o python instalado.

Pasos:

    Coloca el archivo scdeal.tar.gz descargado dentro de la carpeta de tu proyecto scDEAL.

    Descomprime y activa el entorno:

# Crear el directorio donde se desempaquetará el entorno
mkdir -p scDEALenv

# Descomprimir el entorno dentro del directorio
tar -xzf scDEAL.tar.gz -C scDEALenv

# Activar el entorno (esto añade scDEALenv/bin al PATH)
source scDEALenv/bin/activate

## Preparación de Datos

### Descarga de datos

Después de configurar el directorio principal, es necesario descargar los demás recursos requeridos para la ejecución. Por favor, crea y descarga el conjunto de datos en formato zip desde el siguiente enlace: [scDEAL.zip](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/ER2m5OXpYrdPngoAf06pqDoBsiuItm9yvAqg_CjHhNvKSA?e=ckLJ91)

- [Haz clic aquí para descargar scDEAL.zip desde OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/ER2m5OXpYrdPngoAf06pqDoBsiuItm9yvAqg_CjHhNvKSA?e=ckLJ91)
- [Haz clic aquí para descargar scDEAL.zip desde Google Drive](https://drive.google.com/file/d/14mSE1GMi8N8BEt_3MQJSfQvMqg5PH5wI/view?usp=sharing)

El archivo `scDEAL.zip` incluye todos los conjuntos de datos que hemos utilizado en las pruebas.  
Por favor, extrae el archivo zip y coloca el subdirectorio `data` en el directorio raíz de la carpeta `scDEAL`.

|               |     Author             |     Drug         |     GEO access    |     Cells    |     Species           |     Cancer type                        |
|---------------|------------------------|------------------|-------------------|--------------|-----------------------|----------------------------------------|
|     Data 1&2  |     Sharma, et al.     |     Cisplatin    |     GSE117872     |     548      |     Homo   sapiens    |     Oral   squamous cell carcinomas    |
|     Data 3    |     Kong, et al.       |     Gefitinib    |     GSE112274     |     507      |     Homo   sapiens    |     Lung   cancer                      |
|     Data 4    |     Schnepp, et al.    |     Docetaxel    |     GSE140440     |     324      |     Homo   sapiens    |     Prostate   Cancer                  |
|     Data 5    |     Aissa, et al.      |     Erlotinib    |     GSE149383     |     1496     |     Homo sapiens      |     Lung cancer                        |
|     Data 6    |     Bell, et al.       |     I-BET-762    |     GSE110894     |     1419     |     Mus   musculus    |     Acute   myeloid leukemia           |

## Estructura de archivos después de descomprimir

El archivo `scDEAL.zip` también incluye los modelos preentrenados dentro del directorio `save`. Procede a descomprimir `scDEAL.zip`:

# Descomprimir scDEAL.zip en el directorio `scDEAL`
unzip scDEAL.zip

# Ver el contenido del directorio
ls -a


### Contenido de los directorios

Las carpetas del paquete almacenan los siguientes contenidos:

- **root (directorio raíz)**: scripts de Python para ejecutar el programa y el archivo `README.md`.
- **data**: conjuntos de datos necesarios para el entrenamiento.
- **save/logs**: archivos de registro y errores que documentan el estado de ejecución.
- **save/figures & figures**: figuras generadas durante la ejecución.
- **save/models**: modelos entrenados durante la ejecución.
- **save/adata**: resultados exportados en formato AnnData.
- **DaNN**: scripts de Python que describen el modelo.
- **scanpypip**: scripts de utilidades en Python.