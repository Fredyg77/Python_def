import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, mutual_info_score, adjusted_rand_score
import pandas as pd
from typing import List, Tuple

def evaluacion_clusters_kmeans(
    df_X: pd.DataFrame, 
    K_max: int, 
    y: List[int]
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Calcula y grafica métricas de evaluación para clustering:
    homogeneidad, información mutua, Rand ajustado e inercia.

    Parámetros:
    - df_X: DataFrame de características.
    - K_max: Número máximo de clusters a evaluar.
    - y: Etiquetas reales.

    Retorna:
    - k_vals: lista de valores de k.
    - inertias: lista de inercias.
    - homogeneity_indices: lista de índices de homogeneidad.
    - mutual_indices: lista de índices de información mutua.
    - rand_indices: lista de índices de Rand ajustado.
    """
    homogeneity_indices = []
    mutual_indices = []
    rand_indices = []
    inertias = []

    k_vals = list(range(2, K_max + 1))

    for k in k_vals:
        model_KMeans = KMeans(n_clusters=k, random_state=32)
        clusters_n = model_KMeans.fit_predict(df_X)
        
        # Métricas de evaluación
        homogeneity_indices.append(homogeneity_score(y, clusters_n))
        mutual_indices.append(mutual_info_score(y, clusters_n))
        rand_indices.append(adjusted_rand_score(y, clusters_n))
        inertias.append(model_KMeans.inertia_)

    # Gráficos
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    
    sns.lineplot(x=k_vals, y=homogeneity_indices, marker="o", color="steelblue", ax=axes[0, 0])
    axes[0, 0].set_title("Índice de homogeneidad", fontsize=10)
    axes[0, 0].set_xlabel("Número de clusters (k)", fontsize=9)
    axes[0, 0].tick_params(axis='x', labelsize=8)
    axes[0, 0].tick_params(axis='y', labelsize=8)
    axes[0, 0].set_ylabel("")

    sns.lineplot(x=k_vals, y=mutual_indices, marker="o", color="darkorange", ax=axes[0, 1])
    axes[0, 1].set_title("Índice de información mutua", fontsize=10)
    axes[0, 1].set_xlabel("Número de clusters (k)", fontsize=9)
    axes[0, 1].tick_params(axis='x', labelsize=8)
    axes[0, 1].tick_params(axis='y', labelsize=8)
    axes[0, 1].set_ylabel("")

    sns.lineplot(x=k_vals, y=rand_indices, marker="o", color="seagreen", ax=axes[1, 0])
    axes[1, 0].set_title("Índice de Rand ajustado", fontsize=10)
    axes[1, 0].set_xlabel("Número de clusters (k)", fontsize=9)
    axes[1, 0].tick_params(axis='x', labelsize=8)
    axes[1, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].set_ylabel("")

    sns.lineplot(x=k_vals, y=inertias, marker="o", color="purple", ax=axes[1, 1])
    axes[1, 1].set_title("Inercia", fontsize=10)
    axes[1, 1].set_xlabel("Número de clusters (k)", fontsize=9)
    axes[1, 1].tick_params(axis='x', labelsize=8)
    axes[1, 1].tick_params(axis='y', labelsize=8)
    axes[1, 1].set_ylabel("")

    plt.tight_layout()
    plt.show()

    return k_vals, inertias, homogeneity_indices, mutual_indices, rand_indices