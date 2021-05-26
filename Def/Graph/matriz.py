def MatCor (df):
  import pandas as pd
  import numpy as np
  import seaborn as sns
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
