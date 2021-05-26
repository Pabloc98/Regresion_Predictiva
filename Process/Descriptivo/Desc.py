import pandas as pd
import numpy as np

def ProcessDesc(df):  
  """
  Esta función genera el proceso descriptivo del dfset que se ingresa.

  Parametros
  ------------
  df: pandas dfframe
      dfframe
  
  Returns
  ------------
  Head(10)
      Los primeros 10 registros del df
  Info
      Columnas con su respectivo tipo, cantidad de registros no nulos y la cantidad de registros
  Categorias
      Columnas con su respectivo numero de valores unicos por columna
  Resumen
      Estadisticos basicos de cada columna del df
  """
  pd.set_option("max_columns",len(df.columns))

  print("----------------------------------------------------------------------------------------\n --------------------------------HEAD--------------------------------------------\n----------------------------------------------------------------------------------------")
  print(df.head(10))

  print("----------------------------------------------------------------------------------------\n --------------------------------INFO--------------------------------------------\n----------------------------------------------------------------------------------------")
  print(df.info())

  print("----------------------------------------------------------------------------------------\n -----------------------------Categorias-----------------------------------------\n----------------------------------------------------------------------------------------")
  print("Columna, #Categorias")
  for column in list(df.columns):
    print("{}, {}".format(column,len(df[column].value_counts())))

  print("----------------------------------------------------------------------------------------\n ------------------------------Resumen-------------------------------------------\n----------------------------------------------------------------------------------------")
  print(df.describe())

def categorias(df, lista=None):
  """
  Si no se ingresa el parametro lista, esta función generara una serie por cada columna donde indicara el conteo de valores unicos. Por el contrario, si
  se ingresa este parametro, solo generara la serie por cada columna indicada en el parametro lista

  Parametros
  ------------
  df: pandas dfframe
    dfframe
  lista: Lista
    Lista con el nombre de columnas para realizar el conteo de registros unicos

  Returns
  ------------
  Series
    Genera una serie por cada columna seleccionada con el conteo de registros unicos.
  """
  if lista == None:
    for column in list(df.columns):
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Columna: {}".format(column))
        print("{}".format(df[column].value_counts()))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

  else:
    df = df[lista].copy()
    for column in list(df.columns):
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Columna: {}".format(column))
        print("{}".format(df[column].value_counts()))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
