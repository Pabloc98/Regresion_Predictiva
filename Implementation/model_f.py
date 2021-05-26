from Process.Models.models import models
import Implementation.model as  model_pl
from sklearn.model_selection import train_test_split
import numpy as np


def model_f(x,y, modelos, param_modelos="param_dict", escala="escala_list", transformacion="transf_list"):
    """
    Evalua todas las combinaciones posibles arrojando un ranking de los mejores modelos

    Parametros
    -----------------
    x: Array de numpy
        Contiene los valores de las variables independientes del modelo
    y: Array de numpy
        Contiene los valores de la variable dependiente o objetivo
    modelos: lista de strings
        Contiene el nombre exacto de los modelos que se desean evaluar en el ciclo de GridSearchCV
    param_models: diccionario
        Diccionario que abre la opción de cambiar los parámetros establecidos por los desarrolladores (default), para pasar a evaluar más (o menos) combinaciones de hiperparámetros en el GridSearcCV
    escala: lista de strings
        Contiene el nombre exacto de los metodos de escalamiento de sklearn (ej. "minmax")
    transformacion: lista de strings
        Contiene el nombre exacto de los metodos de transformación a la variable dependiente (ej. "Box-Cox")

    Returns
    -----------------
    Dataframe: 
        Imprime un Dataframe con los score y datos relevantes de los modelos evaluados, así como una columna que contiene la lista de valores predecidos.
    Y_test: 
        Array de numpy con los valores reales del conjunto de prueba de los datos
    Y_predict: 
        Array de numpy con los valores predichos con el mejor modelo (Modelo con Valor de 1 en la columna "rank_test_score") del dataframe de resultados
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=33,
    )
    if escala=="escala_list":
        escala1 = ["minmax", "StandardScaler"]
    else:    
        escala1=escala

    if transformacion=="transf_list":
        transformacion1 = []
    else:
        transformacion1 = transformacion

    # Inputs
    seed = 0
    models_dict = {
        "modelos": modelos, #"SGDRegressor","RandomForestRegressor","GaussianProcessRegressor","KNeighborsRegressor","MLPRegressor", "GradientBoostingRegressor"
        "escala": escala1,# "minmax", "StandardScaler"
        "transformacion": transformacion1,# "prueba"
        "cv_list": [] #"KFold","ShuffleSplit"
    }

    if param_modelos == "param_dict":
        # Diccionario de parametros para el gridsearch
        parameters_dict_models = {
            "SGDRegressor":[
                            
                            {
                                "penalty": ["none"],
                            },
                            {
                                "penalty": ["l2"],
                                "alpha": [0.00001, 0.00002, 0.00003],
                            },
                            {
                                "penalty": ["l1"],
                                "alpha": [
                                    0.00001,
                                    0.00002,
                                    0.00003,
                                ],
                                "l1_ratio": [
                                    0.10,
                                    0.15,
                                    0.20,
                                ],
                            }
                        ],
                        
            "RandomForestRegressor":[ # Con los dos parametros su ejecución es de 2 min              
                        {
                            "criterion": ["mse", "mae"],
                            "n_jobs": [-1],
                            "random_state": [seed],
                            
                        },
                        {
                            "criterion": ["mse", "mae"],
                            "n_jobs": [-1],
                            "random_state": [seed],
                            "ccp_alpha": [0.2]
                        }                
                        ],

            "GaussianProcessRegressor":[
                    
                        {
                            "alpha":  [1e-3,1e-10]
                            #"kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]  Si se activa tiene tiempos muy largos
                        },
                        ],

            "KNeighborsRegressor":[                    
                            {
                                "n_neighbors": [5,10,15]
                            },
                            {
                                "n_neighbors": [5,15],
                                "weights": ["uniform", "distance"],
                            },
                            {
                                "weights": ["uniform", "distance"],
                                "leaf_size": [40, 50]                
                            }
                        ],
                        
            "MLPRegressor":[                    
                            {
                                "alpha": [0.0001,0.00001,0.000001],
                                "random_state":[seed]
                            },
                            {
                                "alpha": [0.0001,0.00001,0.000001],
                                "learning_rate": ["adaptive", "invscaling"],
                                "random_state":[seed]
                            },
                            {
                                "learning_rate": ["adaptive", "invscaling"],
                                "solver": ["adam","sgd"],
                                "epsilon": [1e-8, 1e-9],
                                "random_state":[seed]                
                            }
                        ],
            "GradientBoostingRegressor":[
                            {
                                "loss": ['ls', 'lad', 'huber', 'quantile']
                            },
                            {
                                "n_estimators":[10,20,50,70]
                            },
                            {
                                "random_state":[seed]
                            }
                        ],            
        }
    else:
        parameters_dict_models = param_modelos

    data_models  = model_pl.iter_models(models_dict,parameters_dict_models, X_train, Y_train, X_test, Y_test)

    Y_predict = np.array(eval(data_models.loc[0]['Y_predict']))

    return data_models, Y_test, Y_predict