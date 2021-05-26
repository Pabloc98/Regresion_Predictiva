import pandas as pd
import numpy as np
import Def.Graph.Func_graph as gp
import Def.Tf.transform as trf
import Process.Descriptivo.Desc as desc
import Process.Cleaning.Clean as cl
import Process.Models.utils as utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import Def.Tf.fwd_bwd as fb
import Def.Graph.pairplot as pp

def descriptiva():
    datos = str(input('Inserte el nombre exacto del archivo que contiene los datos incluyendo su extensión: '))
    path = "/content/drive/MyDrive/Proyecto_Predictiva/Regresion_Predictiva/Data/Raw/{}".format(datos)
    data = pd.read_excel(path)
    data.head()

    if datos == 'recaudoZER.xlsx':
        # Limpieza del dataset. Esta limpieza esta sobre el archivo recaudoZer que se encuentra en la carpeta Data/Raw
        data_clean = cl.clean_df(data)
    elif datos == 'comparendos_2020.xls':
        data_clean = data[['INFRACCION', 'INMOVILIZADO', 'TIPOVEHI', 'TIPOCOMP', 'JORNADA', 'MES', 'DIAMES', 'DIASEM', 'SEMANA', 'DIA_AÑO', 'VALOR']]
    elif datos == 'acumulada.xlsx':
        data_clean = data[['Minutos', 'Zona', 'dia mes', 'dia semana', 'semana', 'ano', 'Causa', 'Valor']]

    #Corre todo el proceso descriptivo de la data cargada
    desc.ProcessDesc(data_clean)

    #Presentación de graficos de apoyo para el analisis de la data
    print("---------------------Graficos-------------------------------------------------------")
    print("--------------------Matriz Corr-----------------------------------------------------")
    gp.MatCor(data_clean)
    print("---------------------BoxPlot--------------------------------------------------------")
    gp.BoxPlot(data_clean)

    #Generacion de columnas dummys(trf.gen_dummys) o de niveles(trf.gen_niveles) segun la funcion ingresada para la transformacion de datos
    trf.gen_niveles(data_clean)

    # Selección de la columna target
    target = [str(input('Inserte el nombre exacto de la columna objetivo (variable dependiente): '))]
    columnas = [columna for columna in data_clean.columns.to_list() if columna not in target]

    # Separacion de variables dependientes e independientes
    y = data_clean[target[0]].copy().to_numpy()
    x = data_clean[columnas].copy().to_numpy()

    # Separacion de conjuntos de entrenamiento y test

    X_train, X_test, Y_train, Y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=33,
    )

    #Selección de variables que componen el modelo con estrategias forward y backward 
    # (Pendiente mirar forma automática para elegir la mejor opción entre los dos, se hace con backward inicialmente)
    col_fwd = target + columnas
    dataf = data_clean[col_fwd]
    indices, columnas1 = fb.selec_backward(target, columnas, dataf)
    columnas_opt = target + columnas1
    data_opt = data_clean[columnas_opt]

    y = data_clean[target].copy().to_numpy()
    x = data_clean[columnas1].copy().to_numpy()

    print(pp.pairplot(data_opt))

    return x, y