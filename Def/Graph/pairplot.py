
import seaborn as sns
def pairplot(df):
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
    
    sns.pairplot(df)
