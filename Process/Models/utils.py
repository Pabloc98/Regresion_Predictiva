from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
seed = 0

def split(tipo="KFold", n_splits=5 ,random_state=seed, n_repeats=10 ,shuffle=True, test_size=None, train_size=None):
  """Esta función realiza la selección del metodo de separación de los datos a partir del tipo de separación que se desea
  ------------
  tipo: string
    Nombre del metodo de separación de los datos
  n_splits: int
    Numero de separaciones, este parametro lo utilizan todos los metodos de división de datos integrados en al funcion
  random_state: int
    Semilla de aletoriedad. Aplica para: KFold, RepeatedStratifiedKFold, ShuffleSplit
  n_repeats: int
    Repeticiones para el metodo de RepeatedStratifiedKFold
  shuffle: bool
    True o False para indicar si se desea que la seleccion sea aleatoria. Aplica para: KFold
  test_size: int o float
    Si se ingresa un int se toma como la cantidad de registros que debe tener el conjunto de test, si es un float se parte por la cantidad registros correspondientes al porcentaje
  train_size: int o float
    Si se ingresa un int se toma como la cantidad de registros que debe tener el conjunto de train, si es un float se parte por la cantidad registros correspondientes al porcentaje

  Returns
  ------------
  cv
    Modelo de selección de sklearn
  """
  if tipo == "KFold":
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
  if tipo == "GroupKFold":
    cv = GroupKFold(n_splits=n_splits)
  if tipo == "RepeatedStratifiedKFold":
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
  if tipo == "ShuffleSplit":
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
  
  return cv 
