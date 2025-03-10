"""
Script per l’analisi del clustering su MNIST mediante GMM (Gaussian Mixture Model)
con covarianza diagonale, riducendo la dimensionalità tramite PCA.

Per ogni combinazione di:
  - PCA dimension: in [2, 4, 6, ..., 200]
  - Numero di cluster k: da 5 a 15

vengono:
  - Applicata la PCA sul dataset.
  - Fit del modello GMM sul dataset ridotto.
  - Misurato il tempo di esecuzione del fitting.
  - Calcolato il clustering (predict).
  - Calcolato il Rand index (confronto tra etichette vere e cluster ottenuti).
  - Registrato altre metriche: numero di iterazioni, convergenza, log-likelihood finale.

Infine, vengono generati e salvati grafici:
  - Metriche (fitting time, Rand index, iterazioni, lower bound) in funzione
    della dimensionalità PCA per i vari k.
  - Per il modello "migliore" (in questo esempio in base al Rand index massimo):
      • Matrice di confusione (con mapping dei cluster alle etichette vere).
      • Scatter plot dei dati con i cluster.
      • Means del GMM ricostruiti in spazio immagine (28x28).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score, confusion_matrix, pair_confusion_matrix
from scipy.spatial import Voronoi, voronoi_plot_2d


def rand_index_score(labels_true, labels_pred):
    n = len(labels_true)
    tn, fp, fn, tp = pair_confusion_matrix(labels_true, labels_pred).ravel()
    return (tp + tn) / (n * (n - 1))


# -------------------------
# Caricamento e preparazione del dataset MNIST
# -------------------------
print("Caricamento del dataset MNIST...")
# Carica MNIST da OpenML (784 features, immagini 28x28)
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float64)
y = mnist.target.astype(int)

# Normalizziamo i pixel (scalando i valori da 0 a 1)
X /= 255.0

# Per velocizzare l'esecuzione, utilizziamo un sottoinsieme (ad esempio i primi 2000 campioni)
n_samples = 2000
X = X[:n_samples]
y = y[:n_samples]

print(f"Dataset ridotto a {n_samples} campioni.")

# -------------------------
# Definizione delle configurazioni per PCA e GMM
# -------------------------
# Lista di dimensionalità per la PCA (da 2 a 200 con step 2)
pca_dims = list(range(2, 201, 10))
# Range del numero di cluster k per il GMM (da 5 a 15 inclusi)
cluster_range = range(5, 16)

# Lista per salvare i risultati per ogni combinazione PCA_dim - k
results = []

# -------------------------
# Loop sulle configurazioni: PCA -> GMM
# -------------------------
for d in pca_dims:
    print(f"\n=== PCA con n_components = {d} ===")
    # Istanzia e applica PCA
    pca = PCA(n_components=d, random_state=42)
    X_pca = pca.fit_transform(X)

    for k in cluster_range:
        print(f"-> GMM con n_components = {k}")
        # Istanzia il GMM con covarianza diagonale
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)

        # Misura il tempo di esecuzione del fitting
        start_time = time.time()
        gmm.fit(X_pca)
        exec_time = time.time() - start_time

        # Ottieni le etichette di cluster per ogni campione
        cluster_labels = gmm.predict(X_pca)

        # Calcola il Rand index
        ri = rand_index_score(y, cluster_labels)

        # Salva le metriche utili
        results.append(
            {
                "pca_dim": d,
                "n_clusters": k,
                "execution_time_sec": exec_time,
                "rand_index": ri,
                "n_iter": gmm.n_iter_,
                "converged": gmm.converged_,
                "lower_bound": gmm.lower_bound_,
            }
        )

        print(
            f"   Tempo esecuzione: {exec_time:.4f} s | Rand index: {ri:.4f} | "
            f"Iterazioni: {gmm.n_iter_} | Convergenza: {gmm.converged_} | "
            f"Lower bound: {gmm.lower_bound_:.2f}"
        )

# Trasforma i risultati in un DataFrame per una migliore visualizzazione
results_df = pd.DataFrame(results)
print("\n=== Risultati Riassuntivi ===")
print(results_df)

# Salva i risultati su file CSV (opzionale)
results_df.to_csv("gmm_mnist_results.csv", index=False)
print("\nRisultati salvati su 'gmm_mnist_results.csv'.")

# -------------------------
# Creazione della cartella per salvare i grafici
# -------------------------
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------
# Grafici: Metriche in funzione della PCA dimension per i vari k
# -------------------------

# 1. Fitting time vs PCA dimension
plt.figure(figsize=(10, 6))
for k in cluster_range:
    subset = results_df[results_df["n_clusters"] == k]
    plt.plot(
        subset["pca_dim"], subset["execution_time_sec"], marker="o", label=f"k={k}"
    )
plt.xlabel("PCA Dimension")
plt.ylabel("Fitting Time (sec)")
plt.title("Fitting Time vs PCA Dimension per k")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "fitting_time.png"), dpi=300)
plt.close()

# 2. Rand index vs PCA dimension
plt.figure(figsize=(10, 6))
for k in cluster_range:
    subset = results_df[results_df["n_clusters"] == k]
    plt.plot(subset["pca_dim"], subset["rand_index"], marker="o", label=f"k={k}")
plt.xlabel("PCA Dimension")
plt.ylabel("Rand Index")
plt.title("Rand Index vs PCA Dimension per k")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "rand_index.png"), dpi=300)
plt.close()

# 3. Numero di iterazioni vs PCA dimension
plt.figure(figsize=(10, 6))
for k in cluster_range:
    subset = results_df[results_df["n_clusters"] == k]
    plt.plot(subset["pca_dim"], subset["n_iter"], marker="o", label=f"k={k}")
plt.xlabel("PCA Dimension")
plt.ylabel("Numero di Iterazioni")
plt.title("Iterazioni vs PCA Dimension per k")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "n_iterations.png"), dpi=300)
plt.close()

# 4. Lower bound vs PCA dimension
plt.figure(figsize=(10, 6))
for k in cluster_range:
    subset = results_df[results_df["n_clusters"] == k]
    plt.plot(subset["pca_dim"], subset["lower_bound"], marker="o", label=f"k={k}")
plt.xlabel("PCA Dimension")
plt.ylabel("Lower Bound")
plt.title("Lower Bound vs PCA Dimension per k")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "lower_bound.png"), dpi=300)
plt.close()

# -------------------------
# Selezione del modello "migliore" in base al Rand index massimo
# -------------------------
best_index = results_df["rand_index"].idxmax()
best_config = results_df.loc[best_index]
best_pca_dim = int(best_config["pca_dim"])
best_k = int(best_config["n_clusters"])

print(
    f"\nModello migliore scelto: PCA dim = {best_pca_dim}, k = {best_k} (Rand index = {best_config['rand_index']:.4f})"
)


# -------------------------
# Funzione per mappare i cluster alle etichette vere
# -------------------------
def map_clusters_to_labels(true_labels, cluster_labels):
    """Per ogni cluster assegna l'etichetta vera più frequente."""
    mapped_labels = np.zeros_like(cluster_labels)
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Trova l'etichetta più comune in questo cluster
            assigned_label = np.bincount(true_labels[mask]).argmax()
            mapped_labels[mask] = assigned_label
    return mapped_labels


# -------------------------
# Calcolo del modello migliore: PCA e GMM con i parametri migliori
# -------------------------
pca_best = PCA(n_components=best_pca_dim, random_state=42)
X_pca_best = pca_best.fit_transform(X)

gmm_best = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=42)
gmm_best.fit(X_pca_best)
cluster_labels_best = gmm_best.predict(X_pca_best)

# Mappatura dei cluster alle etichette vere
mapped_labels = map_clusters_to_labels(y, cluster_labels_best)

# -------------------------
# Grafico: Matrice di confusione (mappata)
# -------------------------
cm = confusion_matrix(y, mapped_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Etichetta Predetta")
plt.ylabel("Etichetta Vera")
plt.title(f"Matrice di Confusione (PCA: {best_pca_dim}, k = {best_k})")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# -------------------------
# Grafico: Scatter plot dei dati con i cluster (per visualizzazione, usiamo le prime 2 componenti)
# -------------------------
# Se il PCA migliore ha almeno 2 componenti, usiamo le prime 2 per visualizzare i dati
if best_pca_dim >= 2:
    X_vis = X_pca_best[:, :2]
else:
    X_vis = X_pca_best

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels_best, cmap="viridis", s=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Clusters (PCA: {best_pca_dim}, k = {best_k})")
plt.colorbar(sc, label="Etichetta Cluster")
plt.savefig(os.path.join(output_dir, "clusters_scatter.png"), dpi=300)
plt.close()

# -------------------------
# Grafico: Means del GMM ricostruiti in spazio immagine (28x28)
# utilizzando la mappatura dei cluster alle etichette reali
# -------------------------

# Calcolare le medie in spazio PCA e poi ricostruiscile nello spazio originale
means_pca_best = gmm_best.means_
means_original_best = pca_best.inverse_transform(means_pca_best)

# Creare una mappatura: per ogni cluster, memorizzare l'etichetta reale più frequente
cluster_to_true = {}
for cluster in np.unique(cluster_labels_best):
    # Trova gli indici dei campioni assegnati a questo cluster
    indices = np.where(cluster_labels_best == cluster)[0]
    # Ottieni la label reale più frequente tra questi campioni
    true_label = np.bincount(y[indices]).argmax()
    cluster_to_true[cluster] = true_label

clusters_sorted = sorted(cluster_to_true.keys(), key=lambda c: cluster_to_true[c])
ordered_means = [means_original_best[c] for c in clusters_sorted]
ordered_true_labels = [cluster_to_true[c] for c in clusters_sorted]

# Creare la figura con tanti subplot quanti sono i cluster mappati
n_plots = len(clusters_sorted)
fig, axes = plt.subplots(1, n_plots, figsize=(2 * n_plots, 2))

# Se c'è un solo asse (caso in cui n_plots == 1), lo mettiamo in una lista
if n_plots == 1:
    axes = [axes]

# Visualizza ogni "mean" con il titolo corrispondente all'etichetta reale
for ax, mean, label in zip(axes, ordered_means, ordered_true_labels):
    ax.imshow(mean.reshape(28, 28), cmap="gray")
    ax.set_title(f"Etich. {label}")
    ax.axis("off")

plt.suptitle(f"Means del GMM mappate (PCA: {best_pca_dim}, k = {best_k})")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gmm_means_best.png"), dpi=300)
plt.close()


# ------------------------------
# CREAZIONE DEL DIAGRAMMA DI VORONOI
# ------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
gmm = GaussianMixture(n_components=14, covariance_type="diag", random_state=42)
gmm.fit(X_pca)


centroids = gmm.means_  # Centroidi dei cluster
vor = Voronoi(centroids)  # Creazione del Voronoi diagram

plt.figure(figsize=(12, 7))

# Disegna il diagramma di Voronoi
voronoi_plot_2d(
    vor,
    show_vertices=False,
    line_colors="k",
    line_width=1.5,
    line_alpha=0.6,
    ax=plt.gca(),
)

# Disegna i punti del dataset con colorazione in base al cluster
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=mapped_labels, cmap="tab10", s=5, alpha=0.6
)

# Disegna i centroidi dei cluster
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="black",
    s=100,
    marker="o",
    edgecolors="white",
    label="Centroids",
)

# ------------------------------
# PERSONALIZZAZIONE DEL GRAFICO
# ------------------------------
plt.title("Voronoi - Gaussian Mixture Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(output_dir, "voronoi.png"), dpi=300)
plt.close()
