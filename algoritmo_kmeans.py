import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def analisis_clusters_kmeans(df_X, K_max=10, random_state=42, show_plot=True):
    """
    Función que combina el método del codo y el coeficiente de silueta
    para evaluar el número óptimo de clusters en un conjunto de datos.
    
    Parámetros:
    - df_X: DataFrame de características numéricas.
    - K_max: número máximo de clusters a evaluar (default: 10).
    - random_state: semilla para reproducibilidad (default: 42).
    - show_plot: bool para mostrar o no los gráficos (default: True).
    
    Retorna:
    - results_df: DataFrame con k, inercia y coeficiente de silueta.
    """
    sum_of_squared_distances = []
    silhouette_coefficients = []
    k_vals = list(range(2, K_max + 1))
    
    for k in k_vals:
        kmeans_model = KMeans(n_clusters=k, random_state=random_state)
        cluster_assignments = kmeans_model.fit_predict(df_X)
        sum_of_squared_distances.append(kmeans_model.inertia_)
        silhouette_coefficients.append(silhouette_score(df_X, cluster_assignments))

    results_df = pd.DataFrame({
        "k": k_vals,
        "inertia": sum_of_squared_distances,
        "silhouette": silhouette_coefficients
        })

    if show_plot:
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.lineplot(x=k_vals, y=sum_of_squared_distances, marker="o", color="steelblue", ax=axes[0])
        axes[0].set_title("Método del Codo: Inercia vs Clusters", fontsize=10)
        axes[0].set_xlabel("Número de clusters (k)", fontsize=9)
        axes[0].set_ylabel("Inercia", fontsize=9)
        axes[0].tick_params(axis='x', labelsize=8)
        axes[0].tick_params(axis='y', labelsize=8)

        sns.lineplot(x=k_vals, y=silhouette_coefficients, marker="o", color="steelblue", ax=axes[1])
        axes[1].set_title("Coeficiente de Silueta vs Clusters", fontsize=10)
        axes[1].set_xlabel("Número de clusters (k)", fontsize=9)
        axes[1].set_ylabel("Coeficiente de Silueta", fontsize=9)
        axes[1].tick_params(axis='x', labelsize=8)
        axes[1].tick_params(axis='y', labelsize=8)

        plt.tight_layout(pad=3.0)
        plt.show()
    
    return results_df