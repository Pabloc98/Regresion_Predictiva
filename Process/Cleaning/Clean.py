def clean_df(df): 
  """
  Esta función realiza la limpieza del dataset ingresado.
  Si se quiere configurar algun proceso para la limpieza de datos del dataset, se debe agregar en este script

  Parametros
  ------------
  df: pandas dataframe
    Dataframe

  Returns
  ------------
  df
    dataframe procesado con la limpieza y transformación de columnas a nivel de negocio, para ser pasado al proceso de transformación y modelación.
  """
  import datetime
  df["Dia"] = df["Fecha inicio"].dt.day
  df["Dia_semana"] = df["Fecha inicio"].dt.weekday.apply(lambda x: x + 1) # Lunes:1... Domingo:7
  df["Hora_aux_ini"] = df["Fecha inicio"].dt.hour #Temporal
  df["Franja_inicio"] = df["Hora_aux_ini"].map(lambda x: 1 if 0 <= x < 12 else (2 if 12 <= x < 18 else 3 )) #Mañana:1, Tarde:2, Noche:3
  df["Hora_aux_fin"] = df["Fecha fin"].dt.hour #Temporal
  df["Franja_fin"] = df["Hora_aux_fin"].map(lambda x: 1 if 0 <= x < 12 else (2 if 12 <= x < 18 else 3 ))#Mañana:1, Tarde:2, Noche:3

  df["Cambio_franja"] = df["Franja_fin"] - df["Franja_inicio"]
  df["Fecha"] = df["Fecha inicio"].dt.date.map(lambda x: x.toordinal())

  remover = ["Codigo de parqueo", 
  "Nombre recaudador", 
  "Documento recaudador", 
  "Placa",
  "Codigo de cobro",
  "Hora_aux",
  "Hora_aux_ini",
  "Hora_aux_fin",
  "Franja_horaria",
  "Cambio de franja",
  "Fecha inicio",
  "Fecha fin",
  "Franja_inicio",
  "Franja_fin",
  "Cambio_franja"]
  columnas = [col for col in list(df.columns) if col not in remover]
 
  return df[columnas].copy()
