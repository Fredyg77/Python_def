
import pandas as pd
import scipy.stats as stats
import numpy as np

def calcular_correlaciones(df):
    """
    Calcula las matrices de correlación de Pearson, Spearman y Kendall
    solo sobre las variables numéricas del DataFrame.
    También calcula los p-valores de las pruebas de hipótesis para cada correlación.

    Parámetro:
    - df: DataFrame de entrada (puede tener columnas no numéricas)

    Retorna:
    - Diccionario con las tres matrices de correlación y los p-valores
    """
    # Seleccionar solo columnas numéricas
    df_numerico = df.select_dtypes(include='number')

    # Inicializar diccionario de resultados
    resultados = {}

    # Correlación de Pearson
    cor_pearson  = df_numerico.corr(method='pearson')
    pval_pearson = pd.DataFrame(index=df_numerico.columns, columns=df_numerico.columns)
    for col1 in df_numerico.columns:
        for col2 in df_numerico.columns:
            if col1 != col2:
                pval_pearson.loc[col1, col2] = round(stats.pearsonr(df_numerico[col1], df_numerico[col2])[1], 2)
            else:
                pval_pearson.loc[col1, col2] = "-"  # Reemplazar NaN con "-"

    # Correlación de Spearman
    cor_spearman = df_numerico.corr(method='spearman')
    pval_spearman = pd.DataFrame(index=df_numerico.columns, columns=df_numerico.columns)
    for col1 in df_numerico.columns:
        for col2 in df_numerico.columns:
            if col1 != col2:
                pval_spearman.loc[col1, col2] = round(stats.spearmanr(df_numerico[col1], df_numerico[col2])[1], 2)
            else:
                pval_spearman.loc[col1, col2] = "-"  # Reemplazar NaN con "-"

    # Correlación de Kendall
    cor_kendall = df_numerico.corr(method='kendall')
    pval_kendall = pd.DataFrame(index=df_numerico.columns, columns=df_numerico.columns)
    for col1 in df_numerico.columns:
        for col2 in df_numerico.columns:
            if col1 != col2:
                pval_kendall.loc[col1, col2] = round(stats.kendalltau(df_numerico[col1], df_numerico[col2])[1], 2)
            else:
                pval_kendall.loc[col1, col2] = "-"  # Reemplazar NaN con "-"

    # Almacenar los resultados en el diccionario
    resultados['pearson'] = {'correlaciones': cor_pearson.round(2), 'pvalores': pval_pearson}
    resultados['spearman'] = {'correlaciones': cor_spearman.round(2), 'pvalores': pval_spearman}
    resultados['kendall']  = {'correlaciones': cor_kendall.round(2), 'pvalores': pval_kendall}

    return resultados

def imprimir_resultados(resultados):
    """
    Imprime las correlaciones y los p-valores solo para la parte inferior de las matrices.
    Incluye una línea de separación entre las matrices.
    """
    for metodo in resultados:
        print(f"Correlación de {metodo.capitalize()}:")
        correlacion = resultados[metodo]['correlaciones']
        pvalores = resultados[metodo]['pvalores']

        # Mostrar solo la parte inferior de la matriz (con la diagonal)
        correlacion_matriz = correlacion.where(np.tril(np.ones(correlacion.shape), k=0).astype(bool))
        pvalores_matriz = pvalores.where(np.tril(np.ones(pvalores.shape), k=0).astype(bool))

        # Reemplazar NaN por espacios vacíos en la matriz de correlación
        correlacion_matriz = correlacion_matriz.fillna(" ")

        # Reemplazar NaN por espacios vacíos y "-" en la diagonal para los p-valores
        pvalores_matriz = pvalores_matriz.fillna(" ").replace({"-": "-"})

        # Imprimir resultados con un espacio de separación
        print(correlacion_matriz.round(2))
        print(" " * 40)  # Línea separadora

        # Agregar título "Matriz de P-values"
        print("Matriz de P-values:")
        print(pvalores_matriz.round(2).replace("nan", "-"))  # Reemplazar NaN por "-"
        print("=" * 40 + "\n")  # Separador entre cada método
