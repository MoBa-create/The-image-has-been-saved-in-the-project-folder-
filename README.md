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

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/MoBa-create/Customer-Segmentation-KMeans-vs-DBSCAN.git
cd Customer-Segmentation-KMeans-vs-DBSCAN
pip install -r requirements.txt

3. Execution
Run the comparison pipeline script:

	python src/segmentation_comparison.py

🛠️ Project Structure

├── assets/                  # Generated plots and visualization assets
├── src/
│   └── segmentation_comparison.py  # Main Python execution script
├── .gitignore               # Files to ignore in Git
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
