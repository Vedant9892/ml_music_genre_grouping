from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

def apply_pca(X, n_components=2, save_path=None):
    """
    Apply PCA for dimensionality reduction
    
    Args:
        X: Feature matrix
        n_components: Number of components (2 or 3)
        save_path: Path to save model (optional)
    
    Returns:
        X_pca: Transformed coordinates
        pca: Fitted PCA model
    """
    print(f"\nApplying PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    total_var = sum(explained_var) * 100
    
    print(f"✓ PCA completed")
    print(f"  Explained variance by component:")
    for i, var in enumerate(explained_var):
        print(f"    PC{i+1}: {var*100:.2f}%")
    print(f"  Total explained variance: {total_var:.2f}%")
    
    if save_path:
        joblib.dump(pca, save_path)
        print(f"  Model saved to {save_path}")
    
    return X_pca, pca

def plot_clusters_2d(X_pca, labels, title="Cluster Visualization", save_path=None):
    """
    Create 2D scatter plot of clusters
    
    Args:
        X_pca: PCA-transformed coordinates (2D)
        labels: Cluster labels
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='tab10',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")
    
    plt.show()
