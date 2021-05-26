import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def BoxPlot(data, remover = None):
  """Esta función realiza un boxplot por cada variable numerica que contenga el dataset y que no este incluida en la lista remover
  Parametros
  ------------
  df: pandas dataframe
      Dataframe
  remover: lista
      Lista que indica las columnas que no seran tenidas en cuenta para la elaboración del grafico
  Returns
  ------------
  BoxPlot
      Gráfico de cajas y bigotes para cada variable
  """
  if remover == None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xticks(rotation=90);
  else:
    columnas = list(data.columns)
    columnas_opt = [col for col in columnas if col not in remover]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[columnas_opt])
    plt.xticks(rotation=90);
    plt.show()


def MatCor (df):
  """Esta función realiza una matriz de correlación por cada variable numerica que contenga el dataset
  Parametros
  ------------
  df: pandas dataframe
      Dataframe
  Returns
  ------------
  Matriz de correlación
      Mapa de calor de una matriz de correlación
  """
  mcor = df.corr()
  sns.heatmap(mcor, annot = True)
  plt.show()
  
def pairplot(df, agregacion):
    """Esta función realiza un pairplot
    
    Parametros
    ------------
    df: pandas dataframe
        Dataframe procesado con columnas que aporten valor
    agregación: columna del df
        Variable por la cual se realizara la agregación para generar los colores del gráfico
    
    Returns
    ------------
    Pairplot
        Gráfico pairplot de dispersiones entre variables y distribuciones en la diagonal principal
    """
    sns.pairplot(df, hue=agregacion)
