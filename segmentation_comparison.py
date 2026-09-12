import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

def generate_synthetic_data(n_samples=300, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        'CustomerID': range(1001, 1001 + n_samples),
        'Annual_Income_k': np.random.randint(15, 140, size=n_samples),
        'Spending_Score': np.random.randint(1, 100, size=n_samples)
    })
    
    outliers = pd.DataFrame({
        'CustomerID': range(2001, 2006),
        'Annual_Income_k': [10, 160, 165, 5, 155],
        'Spending_Score': [5, 98, 2, 95, 50]
    })
    
    return pd.concat([data, outliers], ignore_index=True)

def main():
    print("[1/4] Generating Synthetic Customer Data...")
    df = generate_synthetic_data()
    
    print("[2/4] Preprocessing & Scaling Features...")
    X = df[['Annual_Income_k', 'Spending_Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[3/4] Training K-Means Model (K=5)...")
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, df['KMeans_Cluster'])
    
    print("[4/4] Training DBSCAN Model (eps=0.32, min_samples=5)...")
    dbscan = DBSCAN(eps=0.32, min_samples=5)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
    
    print("\n" + "="*40)
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
    outliers_count = (df['DBSCAN_Cluster'] == -1).sum()
    print(f"DBSCAN Detected Outliers (Noise): {outliers_count} points")
    print("="*40 + "\n")
    
    os.makedirs(ASSETS_DIR, exist_ok=True)
    output_path = os.path.join(ASSETS_DIR, 'comparison_plot.png')
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(
        x='Annual_Income_k', y='Spending_Score', hue='KMeans_Cluster', 
        data=df, palette='Set1', ax=ax[0], s=70
    )
    ax[0].set_title('K-Means Clustering (K=5)', fontsize=13)
    ax[0].grid(True)
    
    sns.scatterplot(
        x='Annual_Income_k', y='Spending_Score', hue='DBSCAN_Cluster', 
        data=df, palette='tab10', ax=ax[1], s=70
    )
    ax[1].set_title('DBSCAN Clustering (Outliers = -1)', fontsize=13)
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ The image has been saved in the project folder:\n   {output_path}\n")
    
    plt.show()

if __name__ == '__main__':
    main()