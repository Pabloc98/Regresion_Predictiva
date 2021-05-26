import matplotlib.pyplot as plt
import seaborn as sns

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
