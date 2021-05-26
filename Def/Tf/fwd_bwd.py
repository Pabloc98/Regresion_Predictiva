import statsmodels.api as sm
import numpy as np

def selec_forward(v_dependiente, v_independientes, df="df"):
  """Realiza la selección de variables explicativas bajo la estrategia forward
  
  Parametros
  ----------------
  
  **v_dependiente**: 
    Si df es un pd.DataFrame: Esta variable recibe una lista de strings con el nombre de la variable dependiente
    exactamente como aparece en el df.
    Si df es None: Si no se introduce ningún dataframe, esta variable recibe un array de numpy con los valores
    de la variable dependiente.
    En cualquier otro caso arroja error.
  
  **v_independientes**:
    Si df es un pd.DataFrame: Esta variable recibe una lista de strings con los nombres de las variables independientes 
    exactamente como aparecen en el df.
    Si df es None: Si no se introduce ningún dataframe, esta variable recibe un array de numpy con los valores de las 
    variables independientes
    En cualquier otro caso arroja error.
  
  **df**: pandas dataframe
    Contiene todas las variables del modelo

    
  **Recordatorio: Ambas variables deben tener la misma clase para el funcionamiento de la función
 
  Returns
  ----------------
  Lista con los indices de las columnas a utilizar en el modelo óptimo
  """
  
  if type(v_dependiente) == type(np.array(1)) and type(v_independientes) == type(np.array(1)): #and df is None:
    X = v_independientes
    Y = v_dependiente

  
  elif type(v_dependiente) == type([1]) and type(v_independientes) == type([1]):
    X = df[v_independientes].to_numpy()
    Y = df[v_dependiente].to_numpy()
  else:
    print('Not supported for v_dependiente or v_independientes class')
  
  col_index = list(range(len(X[0])))

  menorAIC = None
  v_modelo = []
  modelo = []
  #Ciclo para elección de la primera variable:
  for column in range(len(X[0])):
    model = sm.OLS(Y, X[:, column])
    results = model.fit()

    if menorAIC is None or results.aic < menorAIC:
      menorAIC = results.aic
      v_modelo += [column]
      col_index.remove(column)
  continuar = True
  while continuar:
    continuar = False
    for i in col_index:
      model = sm.OLS(Y, X[:, v_modelo + [i]])
      results = model.fit()
      if menorAIC > results.aic:
        menorAIC = results.aic
        v_modelo += [i] 
        col_index.remove(i)
        continuar = True
    
    if len(v_modelo) == len(X[0]):
      continuar = False 

  if type(v_dependiente) == type([1]) and type(v_independientes) == type([1]):
    for i in v_modelo:
      modelo = modelo + [v_independientes[i]]
    return v_modelo, modelo
  else: 
    return v_modelo
  
    
def selec_backward(v_dependiente, v_independientes, df="df"):
  """Realiza la selección de variables explicativas bajo la estrategia backward
  
  Parametros
  ----------------
  
  **v_dependiente**: 
    Si df es un pd.DataFrame: Esta variable recibe una lista de strings con el nombre de la variable dependiente exactamente como aparece en el df.
    Si df es None: Si no se introduce ningún dataframe, esta variable recibe un array de numpy con los valores de la variable dependiente.
    En cualquier otro caso arroja error.
  
  **v_independientes**:
    Si df es un pd.DataFrame: Esta variable recibe una lista de strings con los nombres de las variables independientes exactamente como aparecen en el df.
    Si df es None: Si no se introduce ningún dataframe, esta variable recibe un array de numpy con los valores de las variables independientes
    En cualquier otro caso arroja error.
  
  **df**: pandas dataframe
    Contiene todas las variables del modelo

    
  **Recordatorio: Ambas variables deben tener la misma clase para el funcionamiento de la función
 
  Returns
  ----------------
  Lista con los indices de las columnas a utilizar en el modelo óptimo para el caso de arrays de numpy
  Lista con los indices de las columnas a utilizar y los nombres correspondientes a las columnas para el caso de pd.DataFrame 
  """
  
  if type(v_dependiente) == type(np.array(1)) and type(v_independientes) == type(np.array(1)): #and df is None:
    X = v_independientes
    Y = v_dependiente

  
  elif type(v_dependiente) == type([1]) and type(v_independientes) == type([1]):
    X = df[v_independientes].to_numpy()
    Y = df[v_dependiente].to_numpy()
  else:
    print('Not supported for v_dependiente or v_independientes class')
  
  col_index = list(range(len(X[0])))
  modelo =[]
  menorAIC = None
  v_modelo = []
  continuar = True

  while continuar:

    continuar = False
    
    for i in col_index:

      intento = col_index.copy()
      intento.remove(i)
      model = sm.OLS(Y, X[:, intento])
      results = model.fit()

      if menorAIC is None or menorAIC > results.aic:
        menorAIC = results.aic
        v_modelo = intento.copy()
        continuar = True
    
      if len(v_modelo) == 1:
        continuar = False 
    
    col_index = v_modelo.copy()

  if type(v_dependiente) == type([1]) and type(v_independientes) == type([1]):
    for i in v_modelo:
      modelo = modelo + [v_independientes[i]]
    return v_modelo, modelo
  else: 
    return v_modelo
