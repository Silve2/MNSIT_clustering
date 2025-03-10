"""
Implementazione di Normalized Cut per clustering su MNIST con variazione di PCA e numero di cluster (k).
Salva le metriche in un CSV, produce grafici (inclusa la Voronoi tessellation per la configurazione migliore in 2D)
e visualizza 4 immagini per ogni cluster nella configurazione con il massimo Rand Index.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pair_confusion_matrix  # Per il calcolo del Rand Index
from scipy.spatial import Voronoi, voronoi_plot_2d

# Impostiamo un seed per la riproducibilità
np.random.seed(0)

# ------------------------------
# 0. Impostazioni iniziali
# ------------------------------

# Directory di output per i grafici
output_dir = "normalized_cut_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Funzione per il calcolo del Rand Index secondo:
# R = (TP + TN) / (TP + TN + FP + FN)
def rand_index_score(labels_true, labels_pred):
    tn, fp, fn, tp = pair_confusion_matrix(labels_true, labels_pred).ravel()
    return (tp + tn) / (tp + tn + fp + fn)


# ------------------------------
# 1. Caricamento e Preprocessing del dataset MNIST
# ------------------------------

print("Caricamento del dataset MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float64)  # Immagini in formato vettoriale (784 features)
y = mnist.target.astype(int)

# Per velocità usiamo un sottoinsieme (ad es. 2000 campioni)
n_samples = 2000
X = X[:n_samples]
y = y[:n_samples]

# Standardizziamo i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 2. Parametri da testare
# ------------------------------

# Livelli di riduzione dimensionale tramite PCA:
pca_dims = range(2, 201, 10)
# Numero di cluster da testare (k da 5 a 15)
k_list = list(range(5, 16))

# Lista per salvare le metriche per ciascuna configurazione
results = []

# ------------------------------
# 3. Ciclo sulle configurazioni: PCA e numero di cluster (k)
# ------------------------------

print("Esecuzione del clustering Normalized Cut con varie configurazioni...")
for d in pca_dims:
    print(f"  - Riduzione a {d} dimensioni tramite PCA")
    pca = PCA(n_components=d, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    for k in k_list:
        print(f"      > Normalized Cut con k = {k}")
        start_time = time.time()
        try:
            # Utilizziamo la SpectralClustering (che implementa il criterio normalized cut)
            sc = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=10,
                assign_labels="kmeans",
                random_state=0,
            )
            sc.fit(X_pca)
            labels_pred = sc.labels_
            fit_time = time.time() - start_time
            r_index = rand_index_score(y, labels_pred)
            n_clusters = len(np.unique(labels_pred))
            results.append(
                {
                    "pca_dim": d,
                    "k": k,
                    "n_clusters": n_clusters,
                    "rand_index": r_index,
                    "fit_time": fit_time,
                }
            )
        except Exception as e:
            print(f"         Avviso: errore per PCA={d}, k={k}: {e}")
            results.append(
                {
                    "pca_dim": d,
                    "k": k,
                    "n_clusters": np.nan,
                    "rand_index": np.nan,
                    "fit_time": np.nan,
                }
            )


# Creazione del DataFrame dei risultati e salvataggio in CSV
df_results = pd.DataFrame(results)
csv_filename = "normalized_cut_metrics.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Metriche salvate in {csv_filename}")

# ------------------------------
# 4. Grafici riassuntivi
# ------------------------------

# Grafico: Rand Index vs Numero di Cluster (k) per ciascun livello di PCA
plt.figure(figsize=(8, 6))
for d in pca_dims:
    subset = df_results[df_results["pca_dim"] == d]
    if subset["rand_index"].isnull().all():
        continue
    plt.plot(subset["k"], subset["rand_index"], marker="o", label=f"PCA {d}")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Rand Index")
plt.title("Rand Index vs k (Normalized Cut)")
plt.legend()
plt.grid(True)
rand_index_fig = os.path.join(output_dir, "normalized_cut_rand_index_vs_k.png")
plt.savefig(rand_index_fig)
plt.close()
print(f"Grafico salvato in {rand_index_fig}")

# Grafico: Fitting Time vs Numero di Cluster (k) per ciascun livello di PCA
plt.figure(figsize=(8, 6))
for d in pca_dims:
    subset = df_results[df_results["pca_dim"] == d]
    if subset["fit_time"].isnull().all():
        continue
    plt.plot(subset["k"], subset["fit_time"], marker="o", label=f"PCA {d}")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Fitting Time (s)")
plt.title("Fitting Time vs k (Normalized Cut)")
plt.legend()
plt.grid(True)
fit_time_fig = os.path.join(output_dir, "normalized_cut_fit_time_vs_k.png")
plt.savefig(fit_time_fig)
plt.close()
print(f"Grafico salvato in {fit_time_fig}")

# ------------------------------
# 5. Voronoi Tessellation per la configurazione migliore (in 2D)
# ------------------------------

# Per la visualizzazione in 2D selezioniamo il caso PCA a 2 componenti
df_pca2 = df_results[
    (df_results["pca_dim"] == 2) & (df_results["rand_index"].notnull())
]
if not df_pca2.empty:
    best_idx_pca2 = df_pca2["rand_index"].idxmax()
    best_config_pca2 = df_pca2.loc[best_idx_pca2]
    best_k_pca2 = int(best_config_pca2["k"])
    best_rand_index_pca2 = best_config_pca2["rand_index"]
    print(
        f"Configurazione migliore in 2D: PCA=2, k={best_k_pca2}, Rand Index={best_rand_index_pca2:.3f}"
    )

    # Ricalcoliamo il clustering in 2D con la migliore configurazione
    pca_2 = PCA(n_components=2, random_state=0)
    X_pca2 = pca_2.fit_transform(X_scaled)
    try:
        sc_best_2d = SpectralClustering(
            n_clusters=best_k_pca2,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=0,
        )
        sc_best_2d.fit(X_pca2)
        labels_best_2d = sc_best_2d.labels_
    except Exception as e:
        print(f"Errore nel clustering per la tessellazione Voronoi: {e}")
        labels_best_2d = None

    if labels_best_2d is not None:
        # Calcoliamo i centri dei cluster (media dei punti in ciascun cluster)
        cluster_centers_2d = []
        for cl in np.unique(labels_best_2d):
            cluster_centers_2d.append(X_pca2[labels_best_2d == cl].mean(axis=0))
        cluster_centers_2d = np.array(cluster_centers_2d)

        # Calcoliamo la Voronoi tessellation a partire dai centri
        vor = Voronoi(cluster_centers_2d)

        # Creazione del grafico
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_pca2[:, 0],
            X_pca2[:, 1],
            c=labels_best_2d,
            cmap="viridis",
            s=10,
            alpha=0.6,
        )
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_points=False,
            show_vertices=False,
            line_colors="orange",
            line_width=2,
        )
        ax.set_title(
            f"Voronoi Tessellation (PCA=2, k={best_k_pca2}, Rand Index={best_rand_index_pca2:.3f})"
        )
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        plt.colorbar(scatter, ax=ax, label="Cluster Label")
        voronoi_fig = os.path.join(
            output_dir, "normalized_cut_voronoi_tessellation.png"
        )
        plt.savefig(voronoi_fig)
        plt.close()
        print(f"Grafico Voronoi salvato in {voronoi_fig}")
else:
    print(
        "Nessuna configurazione valida per PCA=2 trovata per generare la tessellazione Voronoi."
    )

# ------------------------------
# 6. Visualizzazione di 4 immagini per ogni cluster nella configurazione con il massimo Rand Index
# ------------------------------

# Individuiamo la configurazione (tra tutte) con il massimo Rand Index
if not df_results["rand_index"].isnull().all():
    best_idx_all = df_results["rand_index"].idxmax()
    best_config_all = df_results.loc[best_idx_all]
    best_pca_dim = int(best_config_all["pca_dim"])
    best_k = int(best_config_all["k"])
    best_rand = best_config_all["rand_index"]
    print(
        f"Configurazione con massimo Rand Index: PCA={best_pca_dim}, k={best_k}, Rand Index={best_rand:.3f}"
    )

    # Ricalcoliamo il clustering con la configurazione migliore (sulla PCA corrispondente)
    pca_best = PCA(n_components=best_pca_dim, random_state=0)
    X_pca_best = pca_best.fit_transform(X_scaled)
    try:
        sc_best = SpectralClustering(
            n_clusters=best_k,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=0,
        )
        sc_best.fit(X_pca_best)
        labels_best = sc_best.labels_
    except Exception as e:
        print(f"Errore nel clustering per la configurazione migliore: {e}")
        labels_best = None

    if labels_best is not None:
        # Visualizzazione: per ogni cluster, visualizziamo 4 immagini (campioni casuali) del dataset originale
        unique_clusters = np.unique(labels_best)
        n_clusters = len(unique_clusters)
        # Creiamo una figura con n_clusters righe e 4 colonne
        fig, axs = plt.subplots(n_clusters, 4, figsize=(12, 3 * n_clusters))
        # Se c'è un solo cluster, forziamo axs ad avere due dimensioni
        if n_clusters == 1:
            axs = np.array([axs])
        for i, cl in enumerate(unique_clusters):
            # Indici degli esempi appartenenti al cluster cl
            indices = np.where(labels_best == cl)[0]
            # Se il cluster ha almeno 4 immagini, scegliamo 4 campioni casuali; altrimenti mostriamo quelli disponibili
            if len(indices) >= 4:
                sample_indices = np.random.choice(indices, size=4, replace=False)
            else:
                sample_indices = indices
            # Per ciascuna colonna, visualizziamo l'immagine (o lasciamo vuoto se non ci sono abbastanza esempi)
            for j in range(4):
                ax = axs[i, j]
                ax.axis("off")
                if j < len(sample_indices):
                    # Le immagini originali sono vettori di 784 elementi; le rimodelliamo in 28x28
                    img = X[sample_indices[j]].reshape(28, 28)
                    ax.imshow(img, cmap="gray")
                    ax.set_title(f"Cluster {cl}")
                else:
                    ax.set_title("")
        plt.suptitle(
            f"4 immagini per cluster (Configurazione migliore: PCA={best_pca_dim}, k={best_k}, Rand={best_rand:.3f})",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cluster_fig_filename = os.path.join(
            output_dir, "normalized_cut_best_cluster_images.png"
        )
        plt.savefig(cluster_fig_filename)
        plt.close()
        print(
            f"Visualizzazione delle immagini per ogni cluster salvata in {cluster_fig_filename}"
        )
else:
    print(
        "Nessuna configurazione valida trovata per visualizzare le immagini per cluster."
    )

print("Elaborazione completata.")

# Supponiamo che 'labels_best' siano le etichette ottenute dalla configurazione migliore con Spectral Clustering
# e che 'X_pca_best' sia la rappresentazione in PCA (calcolata con pca_best)
unique_clusters = np.unique(labels_best)

# Calcola i centroidi in spazio PCA come media dei campioni appartenenti a ciascun cluster
centroids_pca = []
for cluster in unique_clusters:
    indices = np.where(labels_best == cluster)[0]
    centroid = X_pca_best[indices].mean(axis=0)
    centroids_pca.append(centroid)
centroids_pca = np.array(centroids_pca)

# Trasforma i centroidi nello spazio originale
centroids_original = pca_best.inverse_transform(centroids_pca)

# Se desideri associare ad ogni cluster la label reale più frequente (come nel codice GMM)
cluster_to_true = {}
for cluster in unique_clusters:
    indices = np.where(labels_best == cluster)[0]
    true_label = np.bincount(y[indices]).argmax()
    cluster_to_true[cluster] = true_label

# Ordina i centroidi in base alla label reale (utile se il numero di cluster è simile al numero di classi, ad esempio 10)
clusters_sorted = sorted(cluster_to_true.keys(), key=lambda c: cluster_to_true[c])
ordered_centroids = [centroids_original[c] for c in clusters_sorted]
ordered_true_labels = [cluster_to_true[c] for c in clusters_sorted]

# Visualizza i centroidi come immagini
n_plots = len(ordered_centroids)
fig, axes = plt.subplots(1, n_plots, figsize=(2 * n_plots, 2))
if n_plots == 1:
    axes = [axes]

for ax, centroid, label in zip(axes, ordered_centroids, ordered_true_labels):
    ax.imshow(centroid.reshape(28, 28), cmap="gray")
    ax.set_title(f"Etich. {label}")
    ax.axis("off")

plt.suptitle(f"Centroidi del clustering (PCA: {best_pca_dim}, k = {best_k})")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectral_centroids.png"), dpi=300)
plt.show()
plt.close()
