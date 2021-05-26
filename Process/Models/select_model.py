from Def.Dynamic_dialogs.select_var import select_var
from Def.Tf.fwd_bwd import selec_backward, selec_forward


def select_model(df, forward=True):
  """Esta función abre cajas de dialogo para seleccionar las variables a evaluar bajo la estrategia forward (por defecto) o backward.

  Parámetros:
  ----------------
  df: DataFrame
    Recibe un dataframe limpio el cual contiene las variables dependiente e independientes.
  forward: Bool
    Recibe un booleano, su valor predeterminado es verdadero e indica que se evaluaran los posibles modelos con las variables indicadas bajo una estrategia forward.
    En caso de ser falso se implementa la estrategia backward

  Retorna:
  ----------------
  Una lista con los indices ordenados de las columnas que componen el mejor modelo y una lista de los nombres de estas columnas
  """
  dependiente, independientes = select_var()
  
  if forward is True:

    v_modelo, modelo = selec_forward(dependiente, independientes, df)
  
  elif forward is False:
    v_modelo, modelo = selec_backward(dependiente, independientes, df)
  
  return v_modelo, modelo
