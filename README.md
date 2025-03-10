# 📌 Unsupervised Learning on MNIST: Clustering Handwritten Digits 🤖📊  

This project explores **unsupervised learning** techniques for clustering handwritten digits from the **MNIST dataset**. Instead of relying on labeled data, I used clustering algorithms combined with **Principal Component Analysis (PCA)** to analyze how well different methods can group digits based on their intrinsic patterns.  

---

## 🔍 Project Overview
- **Dataset:** MNIST (70,000 grayscale images, 28x28 pixels each)
- **Goal:** Compare different clustering algorithms to group similar digits without using labels
- **Dimensionality Reduction:** Applied PCA (from 2 to 200 dimensions) to improve efficiency and performance  
- **Clustering Algorithms Used:**  
  ✅ **Gaussian Mixture Models (GMM)** – Probabilistic approach modeling digits as Gaussian distributions  
  ✅ **Mean Shift** – Density-based clustering that finds the optimal number of clusters  
  ✅ **Normalized Cut (Spectral Clustering)** – Graph-based method identifying clusters through relationships in data  

---

## 📊 Analysis & Findings
- **Clustering performance measured with the Rand Index**
- **Comparison of clustering quality at different PCA dimensions**
- **Evaluation of computational efficiency (training & prediction times)**
- **Visualizations of cluster compositions and mean digit representations**

### 🚀 Key Insights
- **GMM** achieved stable performance, especially at higher PCA dimensions  
- **Mean Shift** adapted well but required more computational power  
- **Normalized Cut** effectively captured complex relationships but needed careful parameter tuning  


