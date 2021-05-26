# Convertir en funcion
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor #as sgd
from sklearn.ensemble import RandomForestRegressor #as rfr
from sklearn.gaussian_process import GaussianProcessRegressor #as gpr no funciona
from sklearn.neighbors import KNeighborsRegressor #as knr
from sklearn.neural_network import MLPRegressor #as mlpr
import pandas as pd
import numpy as np
import Process.Models.utils as utils
from sklearn.ensemble import GradientBoostingRegressor

cv_def = utils.split()

# Parametros a ingresar, modelo y parametros para el modelo
def models(modelo, dic_models,parameters_dict_models, X, Y, X_t, cv=cv_def):

  dic_models[modelo][0] = eval(dic_models[modelo][1])

  clf = GridSearchCV(estimator=dic_models[modelo][0],param_grid=parameters_dict_models[modelo], cv=cv_def)

  clf.fit(X,Y)

  Y_pred = clf.predict(X_t)

  data_summary_model = pd.DataFrame(clf.cv_results_)[pd.DataFrame(clf.cv_results_)["rank_test_score"]==1].head(1)

  data_summary_model.insert(0,"modelo", modelo)

  data_summary_model.reset_index(inplace=True)

  return data_summary_model, Y_pred
