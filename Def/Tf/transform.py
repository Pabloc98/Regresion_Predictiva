from sklearn import preprocessing
import numpy as np
import pandas as pd

def gen_niveles(df, lista=None):
  """Transforma las variables categoricas
  
  Parametros
  ----------------
  df: pandas dataframe
    Contiene las variables categoricas y demas variables del modelo
  
  lista: 
    Lista que contiene los nombres de las columnas categoricas a transformar
  Returns
  ----------------
  Dataframe
    Retorna el dataframe original con columnas adicionales que son la transformación de cada variable en factor
  """
  if lista == None:
    tipos_dic = df.columns.to_series().groupby(df.dtypes).groups
    lista = tipos_dic[np.dtype("O")].to_list()

    for col in lista:
      encoder = preprocessing.LabelEncoder().fit(df[col])

      df['{}_FACTOR'.format(col)] = encoder.transform(df[col])
      df.drop(col, axis=1, inplace=True)
  else:
    for col in lista:
      encoder = preprocessing.LabelEncoder().fit(df[col])

      df['{}_FACTOR'.format(col)] = encoder.transform(df[col])
      df.drop(col, axis=1, inplace=True)
      
def gen_dummys(df, lista=None):
  """Transforma las variables seleccionadas en dummys
    
  Parametros
  ----------------
  df: pandas dataframe
    Contiene las variables categoricas/dummys y demas variables del modelo
  
  lista: 
    Lista que contiene los nombres de las columnas categoricas/dummys a transformar
  Returns
  ----------------
  Dataframe
    Retorna el dataframe original con columnas adicionales que son la transformación de cada variable en factor
  """
  df1 = df.copy()
  
  if lista == None:
    tipos_dic = df.columns.to_series().groupby(df.dtypes).groups
    lista = tipos_dic[np.dtype("O")].to_list()
  
    for col in lista:
      dummy_col = pd.get_dummies(df[col])
      frames = [df1, dummy_col]
      df1 = pd.concat(frames,axis=1)
      df1.drop(col, axis=1, inplace=True)
  else:
     for col in lista:
      dummy_col = pd.get_dummies(df[col])
      frames = [df1, dummy_col]
      df1 = pd.concat(frames,axis=1)
      df1.drop(col, axis=1, inplace=True)
  return df1  

def StandardScaler (df):
 
  """Esta función realiza la transformación estandar para las variables numéricas del dataset

  Parametros
  ------------
  df: pandas dataframe
      Dataframe

  Returns
  ------------
  Variables numéricas del dataset con transformación estandar
  """
  dfnew = df.select_dtypes(include=[int, float]).copy()
  escalador = preprocessing.StandardScaler().fit(dfnew)
  return escalador.transform(dfnew)

def minmax (df):

  """Esta función realiza la transformación de los datos paras que estos queden en una escala del 0 al 1 
  para las variables numéricas del dataset

  Parametros
  ------------
  df: pandas dataframe
      Dataframe

  Returns
  ------------
  Variables numéricas del dataset con transformación estandar
  """
  dfnew = df.select_dtypes(include=[int, float]).copy()
  min_max_scaler = preprocessing.MinMaxScaler()
  df_minmax = min_max_scaler.fit_transform(dfnew)
  return df_minmax

def maxabs (df):

  """Esta función lleva a que cada característica tenga un valor máximo de 1.0

  Parametros
  ------------
  df: pandas dataframe
      Dataframe

  Returns
  ------------
  Variables numéricas del dataset con transformación estandar
  """
  dfnew = df.select_dtypes(include=[int, float]).copy()
  max_abs_scaler = preprocessing.MaxAbsScaler()
  X_train_maxabs = max_abs_scaler.fit_transform(dfnew)
  return X_train_maxabs

def QuantileTransformer (df):

  """Realiza una transformación no lineal que mapea los datos a una 
  distribucion uniforme en el rango 0-1

  Parametros
  ------------
  df: pandas dataframe
      Dataframe

  Returns
  ------------
  Variables numéricas del dataset con transformación estandar
  """
  dfnew = df.select_dtypes(include=[int, float]).copy()
  quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
  X_train_trans = quantile_transformer.fit_transform(dfnew)
  return X_train_trans

def PowerTransformer (df):

  """# Transforma los datos para hacerlos más similares a una distribución 
  normal

  Parametros
  ------------
  df: pandas dataframe
      Dataframe

  Returns
  ------------
  Variables numéricas del dataset con transformación estandar
  """
  dfnew = df.select_dtypes(include=[int, float]).copy()
  pt = preprocessing.PowerTransformer(method="box-cox", standardize=False)
  X_lognormal = np.random.RandomState(616).lognormal(size=(len(dfnew), len(dfnew.columns)))
  return X_lognormal

