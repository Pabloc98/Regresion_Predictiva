def gen_niveles(df, lista):
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
  from sklearn import preprocessing
  for col in lista:
    encoder = preprocessing.LabelEncoder().fit(df[col])

    df['{}_FACTOR'.format(col)] = encoder.transform(df[col])
