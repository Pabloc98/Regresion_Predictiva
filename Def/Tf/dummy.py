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
  
#Pendiente si sobre-escribir el df original o como es el correcto tratamiento en este caso
