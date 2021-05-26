from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, DotProduct
import itertools
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor #as sgd
from sklearn.ensemble import RandomForestRegressor #as rfr
from sklearn.gaussian_process import GaussianProcessRegressor #as gpr no funciona
from sklearn.neighbors import KNeighborsRegressor #as knr
from sklearn.neural_network import MLPRegressor #as mlpr
from sklearn.ensemble import GradientBoostingRegressor #as gbr
import Process.Models.utils as utils
import Def.Tf.transform as trf
import Process.Models.models as models
import Def.Metrics.mse_r2 as metrics

def best_model(x,y, model):
    #Semilla
    seed = 0
    X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=33,#seed
    )


    # Metodo de separación de datos . (KFold, )
    cv = utils.split(tipo='KFold')

    # Diccionario de preprocesamiento de datos(escala): https://jdvelasq.github.io/courses/notebooks/sklearn/fundamentals/4-28_Preprocesado_de_datos_en_sklearn.html
    escala = ["StandardScaler", "minmax", "maxabs", "QuantileTransformer","PowerTransformer"]

    # Diccionario de separación de datos: https://jdvelasq.github.io/courses/notebooks/sklearn/fundamentals/4-24_Esquemas_de_particion_de_los_datos.html
    cv_list = ["KFold", "GroupKFold", "RepeatedStratifiedKFold", "ShuffleSplit"]

    # Diccionario de tranformación de variables dependientes e independientes: https://jdvelasq.github.io/courses/notebooks/sklearn/fundamentals/4-25_Transformacion_no_lineal_de_variables.html


    models_dict = {
        "escala": [],
        "cv_list": ["KFold", "GroupKFold"],
        "modelos": ["SGDRegressor","RandomForestRegressor","GaussianProcessRegressor","KNeighborsRegressor","MLPRegressor", "GradientBoostingRegressor"]
    }

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
    modelo = str(model)
    dic_models = {
        "SGDRegressor":"sgd_model",
        "RandomForestRegressor":"rfr_model",
        "GaussianProcessRegressor":"gpr_model",
        "KNeighborsRegressor":"knr_model",
        "MLPRegressor": "mlpr_model",
        "GradientBoostingRegressor": "gbr_model"
    }
    sgd_model = SGDRegressor()
    rfr_model = RandomForestRegressor()
    gpr_model = GaussianProcessRegressor()
    knr_model = KNeighborsRegressor()
    mlpr_model = MLPRegressor()
    gbr_model = GradientBoostingRegressor()

    clf = GridSearchCV(estimator=eval(dic_models[modelo]),param_grid=parameters_dict_models[modelo], cv=cv)

    clf.fit(X_train,Y_train)

    Y_pred = clf.predict(X_test)

    pd.set_option('display.max_columns', None)
    data_summary_model = pd.DataFrame(clf.cv_results_)[pd.DataFrame(clf.cv_results_)["rank_test_score"]==1]
    print(data_summary_model)


    return Y_test, Y_pred


def iter_models(models_dict,parameters_dict_models, X, Y,X_t, Y_t):

  # Se inician las variables para los dataframe
  data_model = None
  data_model_aux = None

  # Diccionarios de necesarios para la correcta función de script
  dic_escala = {
      "StandardScaler":["trf.StandardScaler(pd.DataFrame(X))", "trf.StandardScaler(pd.DataFrame(Y))"],
      "minmax": ["trf.minmax(pd.DataFrame(X))", "trf.minmax(pd.DataFrame(Y))"],
      "maxabs": ["trf.maxabs(pd.DataFrame(X))", "trf.maxabs(pd.DataFrame(Y))"],
      "QuantileTransformer": ["trf.QuantileTransformer(pd.DataFrame(X))", "trf.QuantileTransformer(pd.DataFrame(Y))"],
      "PowerTransformer":["trf.PowerTransformer(pd.DataFrame(X))", "trf.PowerTransformer(pd.DataFrame(Y))"]
  }

  dic_models = {
      "SGDRegressor":["sgd_model","SGDRegressor()"],
      "RandomForestRegressor":["rfr_model","RandomForestRegressor()"],
      "GaussianProcessRegressor":["gpr_model","GaussianProcessRegressor()"],
      "KNeighborsRegressor":["knr_model","KNeighborsRegressor()"],
      "MLPRegressor": ["mlpr_model", "MLPRegressor()"],
      "GradientBoostingRegressor": ["gbr_model", "GradientBoostingRegressor()"]
  }

  

  # Revisa que parametros se estan usando.(escala, cv_list, modelos, transformaciones)
  lista_iters = []
  for param in list(models_dict.keys()):
    if len(models_dict[param]) == 0:
      pass
    else: 
      lista_iters.append(param)
  print(lista_iters)
  # Valida que todos los modelos ingresados esten contenidos en el diccionario de parametros para cada modelo
  val_m = 0


  # Valida si la lista de modelos es mayor a cero, pues es el parametro minimo requerido
  if len(models_dict["modelos"]) > 0:

    # Itera sobre los modelos ingresados
    for indice, modelo in enumerate(models_dict["modelos"]):
      
      # Valida que los modelos ingresados tengan almenos un parametro en el diccionario de parametros para cada modelo
      if modelo in list(parameters_dict_models.keys()):
        if indice == 0:
          lista_iters.remove("modelos")
        else:
          pass
        
        # Si todos los modelos tienen sus respectivos parametros, se inician los condicionales segun la cantidad de parametros a configurar
        # Se itera sobre modelos, porque este es el unico parametro que puede estar configurado solo
        if len(lista_iters) == 0: # Solo se seleccionaron los modelos
          # Se genera la copia del dataframe para luego agregar
          data_model_aux=data_model
          
          # Generación del modelo
          if "cv_list" in lista_iters:
            data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t, cv)
          else:
            data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t)

          # Se generan las predicciones en forma de lista para ser agregadas al dataframe
          y_predict_list = y_predict.tolist()
          y_list = [y[0] if type(y) == list else y for y in y_predict_list]

          # Generación R2 y mse
          mse, r2 = metrics.mse_r2_no_print(Y_t, y_predict)

          # Se agregan campos al df
          data_model.insert(2,"mse", mse)
          data_model.insert(3,"r2", r2)
          data_model.insert(4,"Y_predict", str(y_list))

          # Se agregan los dataframes
          data_model = data_model.append(data_model_aux)

          print(1)
        
        # Se seleccionan los modelos y otro parametro
        elif len(lista_iters) == 1: 
          
          print(2)

          # Se itera sobre cada una de las configuraciones, generando los escalamiento, transformación de variables y el modelo para cada una
          for iter_2 in models_dict[lista_iters[0]]:

            # Se genera la copia del dataframe para luego agregar
            data_model_aux=data_model
            
            print("iter_2: ---")
            if lista_iters[0] == "escala":
              X = eval(dic_escala[iter_2][0])
              Y = eval(dic_escala[iter_2][1])
              print(iter_2)
            elif lista_iters[0] == "transformacion":
              print("transformacion")
              print(iter_2)
            elif lista_iters[0] == "cv_list":
              cv = utils.split(tipo=iter_2)
              print(cv)
            
            # Generación del modelo
            if "cv_list" in lista_iters:
              data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t, cv)
            else:
              data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t)

            # Se generan las predicciones en forma de lista para ser agregadas al dataframe
            y_predict_list = y_predict.tolist()
            y_list = [y[0] if type(y) == list else y for y in y_predict_list]
            
            # Generación R2 y mse
            mse, r2 = metrics.mse_r2_no_print(Y_t, y_predict)
            
            # Se agregan campos al df
            data_model.insert(2,lista_iters[0], iter_2)
            data_model.insert(3,"mse", mse)
            data_model.insert(4,"r2", r2)
            data_model.insert(5,"Y_predict", str(y_list))

            # Se agregan los dataframes
            data_model = data_model.append(data_model_aux)


        # Se seleccionan los modelos y 2 parametros mas
        elif len(lista_iters) == 2: 
          
          print(3)

          # Se itera sobre cada una de las configuraciones, generando los escalamiento, transformación de variables y el modelo para cada una
          for iter_2 in models_dict[lista_iters[0]]:
            print("iter_2: ---")
            # Se genera la copia del dataframe para luego agregar
            data_model_aux=data_model

            if lista_iters[0] == "escala":
              X = eval(dic_escala[iter_2][0])
              Y = eval(dic_escala[iter_2][1])
              print(iter_2)
            elif lista_iters[0] == "transformacion":
              print("transformacion")
              print(iter_2)
            elif lista_iters[0] == "cv_list":
              cv = utils.split(tipo=iter_2)
              print(cv)

            for iter_3 in models_dict[lista_iters[1]]:
              print("iter_3: ---")
              # Se genera la copia del dataframe para luego agregar
              data_model_aux=data_model
              if lista_iters[1] == "escala":
                X = eval(dic_escala[iter_3][0])
                Y = eval(dic_escala[iter_3][1])
                print(iter_3)
              elif lista_iters[1] == "transformacion":
                print("transformacion")
                print(iter_3)
              elif lista_iters[1] == "cv_list":
                cv = utils.split(tipo=iter_3)
                print(cv)

              # Generación del modelo
              if "cv_list" in lista_iters:
                data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t, cv)
              else:
                data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t)
              
              # Se generan las predicciones en forma de lista para ser agregadas al dataframe
              y_predict_list = y_predict.tolist()
              y_list = [y[0] if type(y) == list else y for y in y_predict_list]

              # Generación R2 y mse
              mse, r2 = metrics.mse_r2_no_print(Y_t, y_predict)

              # Se agregan campos al df
              data_model.insert(2,lista_iters[0], iter_2)
              data_model.insert(3,lista_iters[1], iter_3)
              data_model.insert(4,"mse", mse)
              data_model.insert(5,"r2", r2)
              data_model.insert(6,"Y_predict", str(y_list))

              # Se agregan los dataframes
              data_model = data_model.append(data_model_aux)

        # Se seleccionan los modelos y 3 parametros mas
        elif len(lista_iters) == 3: 
          
          print(4)
          
          # Se itera sobre cada una de las configuraciones, generando los escalamiento, transformación de variables y el modelo para cada una
          for iter_2 in models_dict[lista_iters[0]]:
            print("iter_2: ---")

            # Se genera la copia del dataframe para luego agregar
            data_model_aux=data_model

            if lista_iters[0] == "escala":
              X = eval(dic_escala[iter_2][0])
              Y = eval(dic_escala[iter_2][1])
              print(iter_2)
            elif lista_iters[0] == "transformacion":
              print("transformacion")
              print(iter_2)
            elif lista_iters[0] == "cv_list":
              cv = utils.split(tipo=iter_2)
              print(cv)

            for iter_3 in models_dict[lista_iters[1]]:
              print("iter_3: ---")

              # Se genera la copia del dataframe para luego agregar
              data_model_aux=data_model
              
              if lista_iters[1] == "escala":
                X = eval(dic_escala[iter_3][0])
                Y = eval(dic_escala[iter_3][1])
                print(iter_3)
              elif lista_iters[1] == "transformacion":
                print("transformacion")
                print(iter_3)
              elif lista_iters[1] == "cv_list":
                cv = utils.split(tipo=iter_3)
                print(cv)

              for iter_4 in models_dict[lista_iters[2]]:
                print("iter_4: ---")
                
                # Se genera la copia del dataframe para luego agregar
                data_model_aux=data_model
                
                if lista_iters[2] == "escala":
                  X = eval(dic_escala[iter_4][0])
                  Y = eval(dic_escala[iter_4][1])
                  print(iter_4)
                elif lista_iters[2] == "transformacion":
                  print("transformacion")
                  print(iter_4)
                elif lista_iters[2] == "cv_list":
                  cv = utils.split(tipo=iter_4)
                  print(cv)
                
                # Generación del modelo
                if "cv_list" in lista_iters:
                  data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t, cv)
                else:
                  data_model, y_predict = models.models(modelo, dic_models,parameters_dict_models,X, Y, X_t)

                # Se generan las predicciones en forma de lista para ser agregadas al dataframe
                y_predict_list = y_predict.tolist()
                y_list = [y[0] if type(y) == list else y for y in y_predict_list]

                # Generación R2 y mse
                mse, r2 = metrics.mse_r2_no_print(Y_t, y_predict)

                # Se agregan campos al df
                data_model.insert(2,lista_iters[0], iter_2)
                data_model.insert(3,lista_iters[1], iter_3)
                data_model.insert(4,lista_iters[2], iter_4)
                data_model.insert(5,"mse", mse)
                data_model.insert(6,"r2", r2)
                data_model.insert(7,"Y_predict", str(y_list))

                # Se agregan los dataframes
                data_model = data_model.append(data_model_aux)

        # Reinicia indices, elimina columnas de indices y ordena para generar ranking     
        data_model = data_model.sort_values(by="mean_test_score").copy()
        data_model["rank_test_score"] = data_model["mean_test_score"].rank(ascending = 1)
        data_model.reset_index(inplace=True)
        data_model = data_model.drop(["index","level_0"],axis=1).copy()

      # Si el modelo no se encuentra en los parametros, arroja el siguiente mensaje
      else:
        val_m = 1
        print("Faltan los parametros para el modelo {}".format(modelo))

  # Si el parametro ingresado no es un modelo, arroja el siguiente error
  else:
    print("El parametro minimo requerido debe ser un modelo")
  return data_model 
