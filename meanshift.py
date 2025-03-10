"""
Implementazione di Mean Shift per clustering su MNIST con variazione di PCA e bandwidth.
Salva le metriche in un CSV e produce grafici (inclusa la Voronoi tessellation per la configurazione migliore).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pair_confusion_matrix  # usato per calcolare il Rand index
from scipy.spatial import Voronoi, voronoi_plot_2d

# Directory di output per i grafici
output_dir = "mean_shift_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Funzione per il calcolo del Rand Index secondo:
# R = 2(a+b)/(n(n-1))
def rand_index_score(labels_true, labels_pred):
    """
    Calcola il Rand index a partire dai vettori delle etichette vere e predette.
    Utilizza pair_confusion_matrix per estrarre i conteggi (TP, TN, FP, FN).
    """
    tn, fp, fn, tp = pair_confusion_matrix(labels_true, labels_pred).ravel()
    return (tp + tn) / (tp + tn + fp + fn)


# =========================
# 1. Caricamento e Preprocessing del dataset MNIST
# =========================

print("Caricamento del dataset MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float64)
y = mnist.target.astype(int)

# Per velocità, usiamo un sottoinsieme (ad es. 2000 campioni)
n_samples = 2000
X = X[:n_samples]
y = y[:n_samples]

# Standardizziamo i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2. Parametri da testare
# =========================

# Livelli di riduzione dimensionale: da 2 a 200 componenti (modifica la lista se necessario)
pca_dims = range(2, 201, 10)

# Valori di bandwidth per il Mean Shift (modifica la lista in base al dataset e sperimentazioni)
bandwidth_list = [0.5, 1, 2, 3, 4, 5, 10, 15, 20]

# Lista in cui salvare le metriche per ciascuna configurazione
results = []

# =========================
# 3. Ciclo sulle configurazioni: PCA e bandwidth
# =========================

print("Esecuzione del clustering Mean Shift con varie configurazioni...")
for d in pca_dims:
    print(f"  - Riduzione a {d} dimensioni tramite PCA")
    pca = PCA(n_components=d)
    X_pca = pca.fit_transform(X_scaled)

    for bw in bandwidth_list:
        print(f"      > MeanShift con bandwidth = {bw}")
        start_time = time.time()
        success = False

        # Prova con bin_seeding=True
        try:
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            ms.fit(X_pca)
            success = True
        except ValueError as e:
            print(
                f"         Avviso: errore con bin_seeding=True per PCA={d}, bandwidth={bw}: {e}"
            )
            # Prova con bin_seeding=False
            try:
                ms = MeanShift(bandwidth=bw, bin_seeding=False)
                ms.fit(X_pca)
                success = True
            except ValueError as e2:
                print(
                    f"         Avviso: errore anche con bin_seeding=False per PCA={d}, bandwidth={bw}: {e2}"
                )
                success = False

        if success:
            fit_time = time.time() - start_time
            labels_pred = ms.labels_
            n_clusters = len(np.unique(labels_pred))
            r_index = rand_index_score(y, labels_pred)

            results.append(
                {
                    "pca_dim": d,
                    "bandwidth": bw,
                    "n_clusters": n_clusters,
                    "rand_index": r_index,
                    "fit_time": fit_time,
                }
            )
        else:
            # Salva la configurazione con NaN se il clustering non è andato a buon fine
            results.append(
                {
                    "pca_dim": d,
                    "bandwidth": bw,
                    "n_clusters": np.nan,
                    "rand_index": np.nan,
                    "fit_time": np.nan,
                }
            )

# Creazione del DataFrame dei risultati e salvataggio in CSV
df_results = pd.DataFrame(results)
csv_filename = "mean_shift_metrics.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Metriche salvate in {csv_filename}")

# =========================
# 4. Grafici riassuntivi
# =========================

# Grafico: Rand Index vs Bandwidth per ciascun livello di PCA
plt.figure(figsize=(8, 6))
for d in pca_dims:
    subset = df_results[df_results["pca_dim"] == d]
    if subset["rand_index"].isnull().all():
        continue
    plt.plot(subset["bandwidth"], subset["rand_index"], marker="o", label=f"PCA {d}")
plt.xlabel("Bandwidth")
plt.ylabel("Rand Index")
plt.title("Rand Index vs Bandwidth (Mean Shift)")
plt.legend()
plt.grid(True)
rand_index_fig = os.path.join(output_dir, "mean_shift_rand_index_vs_bandwidth.png")
plt.savefig(rand_index_fig)
plt.close()
print(f"Grafico salvato in {rand_index_fig}")

# Grafico: Fitting Time vs Bandwidth per ciascun livello di PCA
plt.figure(figsize=(8, 6))
for d in pca_dims:
    subset = df_results[df_results["pca_dim"] == d]
    if subset["fit_time"].isnull().all():
        continue
    plt.plot(subset["bandwidth"], subset["fit_time"], marker="o", label=f"PCA {d}")
plt.xlabel("Bandwidth")
plt.ylabel("Fitting Time (s)")
plt.title("Fitting Time vs Bandwidth (Mean Shift)")
plt.legend()
plt.grid(True)
fit_time_fig = os.path.join(output_dir, "mean_shift_fit_time_vs_bandwidth.png")
plt.savefig(fit_time_fig)
plt.close()
print(f"Grafico salvato in {fit_time_fig}")

# =========================
# 5. Voronoi Tessellation per la configurazione migliore (in 2D)
# =========================

# Per la visualizzazione in 2D è opportuno considerare il caso PCA a 2 componenti.
# Selezioniamo la configurazione con PCA=2 che ha il miglior Rand index.
df_pca2 = df_results[
    (df_results["pca_dim"] == 2) & (df_results["rand_index"].notnull())
]
if not df_pca2.empty:
    best_idx = df_pca2["rand_index"].idxmax()
    best_config = df_pca2.loc[best_idx]
    best_bandwidth = best_config["bandwidth"]
    best_rand_index = best_config["rand_index"]

    print(
        f"Configurazione migliore in 2D: PCA=2, bandwidth={best_bandwidth}, Rand Index={best_rand_index:.3f}"
    )

    # Ricalcoliamo la clusterizzazione in 2D con il best bandwidth
    pca_2 = PCA(n_components=2)
    X_pca2 = pca_2.fit_transform(X_scaled)
    try:
        ms_best = MeanShift(bandwidth=best_bandwidth, bin_seeding=True)
        ms_best.fit(X_pca2)
    except ValueError:
        ms_best = MeanShift(bandwidth=best_bandwidth, bin_seeding=False)
        ms_best.fit(X_pca2)
    labels_best = ms_best.labels_
    cluster_centers = ms_best.cluster_centers_

    # Calcolo della Voronoi tessellation a partire dai centri dei cluster
    vor = Voronoi(cluster_centers)

    # Creazione del grafico
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter dei dati colorati in base ai cluster
    scatter = ax.scatter(
        X_pca2[:, 0], X_pca2[:, 1], c=labels_best, cmap="viridis", s=10, alpha=0.6
    )
    # Disegno della Voronoi tessellation (senza mostrare i vertici)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_points=False,
        show_vertices=False,
        line_colors="orange",
        line_width=2,
    )

    ax.set_title(
        f"Voronoi Tessellation (PCA=2, Bandwidth={best_bandwidth}, Rand Index={best_rand_index:.3f})"
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    plt.colorbar(scatter, ax=ax, label="Cluster Label")
    voronoi_fig = os.path.join(output_dir, "mean_shift_voronoi_tessellation.png")
    plt.savefig(voronoi_fig)
    plt.close()
    print(f"Grafico Voronoi salvato in {voronoi_fig}")
else:
    print(
        "Nessuna configurazione valida per PCA=2 trovata per generare la tessellazione Voronoi."
    )

print("Elaborazione completata.")


# stamapare i cluster centers del miglior modello mean_shift con rand_index più alto
print("Cluster centers del miglior modello Mean Shift")
fig, ax = plt.subplots(figsize=(8, 6))
