# Customer Segmentation: K-Means vs. DBSCAN Clustering Comparison

This repository provides an end-to-end Machine Learning implementation comparing two popular clustering algorithms: **K-Means** (Centroid-based) and **DBSCAN** (Density-based). 

The goal is to analyze customer purchasing behavior, handle outliers/noise effectively, and extract actionable business personas.

---

## 📌 Project Overview

Customer segmentation is essential for targeted marketing and customer relationship management (CRM). This project demonstrates:
1. **Data Preprocessing & Scaling**: Standardizing features using `StandardScaler`.
2. **Optimal Cluster Selection**: Utilizing WCSS (Elbow Method) and Silhouette Analysis.
3. **Model Training & Comparison**:
   - **K-Means**: Forces every data point into $K$ clusters (spherical shapes).
   - **DBSCAN**: Identifies arbitrary-shaped clusters and isolates outliers (noise detection).
4. **Business Insight Extraction**: Segmenting customers by Annual Income and Spending Score.

---

## 📊 Key Algorithm Comparison

| Feature | K-Means | DBSCAN |
| :--- | :--- | :--- |
| **Clustering Type** | Partitioning / Centroid-based | Density-based |
| **Cluster Shape** | Spherical / Convex | Arbitrary shapes |
| **Outlier Sensitivity** | High (Forces outliers into clusters) | Low (Isolates outliers as `-1`) |
| **Hyperparameters** | Number of clusters ($K$) | Radius (`eps`), Min Samples (`min_samples`) |
| **Best Used When** | Clean data with defined $K$ | Noisy data with irregular clusters |
