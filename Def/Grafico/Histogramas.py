import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas

def histograma (  ):
  """Esta función realiza un diagrama de caja por cada variable numérica que contiene el conjunto de datos y que no está incluido en la lista 

  Parametros
  ------------
  df: marco de datos de pandas
      Marco de datos
  removedor: lista
      Lista que indica las columnas que no seran tenidas en cuenta para la elaboración del grafico
  Devoluciones
  ------------
  BoxPlot
      Gráfico de cajas y bigotes para cada variable
  """
        
        def histogramas(df):
          nombres = df.columns
          plt.style.use('fivethirtyeight')
          fig, axs = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(20, 6));
          plt.subplots_adjust(wspace = 0.05, hspace=0.1)
          for i in nombres:
            plt.title(i)
            plt.hist(df[i], bins = 60)
            plt.grid(True)
            plt.show()
