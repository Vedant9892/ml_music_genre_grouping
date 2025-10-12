<<<<<<< HEAD
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib

def apply_kmeans(X, n_clusters=10, save_path=None):
    """
    Apply K-Means clustering
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        save_path: Path to save model (optional)
    
    Returns:
        model: Trained KMeans model
        labels: Cluster labels
        score: Silhouette score
    """
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    
    print(f"✓ K-Means completed")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score: {score:.3f}")
    
    if save_path:
        joblib.dump(kmeans, save_path)
        print(f"  Model saved to {save_path}")
    
    return kmeans, labels, score

def apply_gmm(X, n_components=10, save_path=None):
    """
    Apply Gaussian Mixture Model clustering
    
    Args:
        X: Feature matrix
        n_components: Number of components
        save_path: Path to save model (optional)
    
    Returns:
        model: Trained GMM model
        labels: Cluster labels
        probabilities: Probability matrix
        score: BIC score
    """
    print(f"\nTraining GMM with {n_components} components...")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    
    gmm.fit(X)
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    score = gmm.bic(X)
    
    print(f"✓ GMM completed")
    print(f"  BIC Score: {score:.2f}")
    print(f"  Converged: {gmm.converged_}")
    
    if save_path:
        joblib.dump(gmm, save_path)
        print(f"  Model saved to {save_path}")
    
    return gmm, labels, probabilities, score
=======
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

class MusicClusterer:
    """Implement K-Means and GMM clustering for music genre grouping."""
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize models
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            max_iter=300,
            random_state=random_state
        )
        
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            max_iter=300,
            random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.pca_3d = PCA(n_components=3, random_state=random_state)
        
    def preprocess_data(self, df):
        """Standardize features and prepare for clustering."""
        # Select 5 features
        feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
        X = df[feature_cols].values
        
        # Standardize (mean=0, std=1)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_cols
    
    def fit_kmeans(self, X):
        """Train K-Means clustering model."""
        print("Training K-Means clustering...")
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        return labels
    
    def fit_gmm(self, X):
        """Train GMM clustering model."""
        print("Training Gaussian Mixture Model...")
        self.gmm.fit(X)
        labels = self.gmm.predict(X)
        probabilities = self.gmm.predict_proba(X)
        return labels, probabilities
    
    def apply_pca(self, X):
        """Apply PCA for 2D and 3D visualization."""
        print("Applying PCA for dimensionality reduction...")
        
        # 2D PCA
        X_pca_2d = self.pca.fit_transform(X)
        variance_2d = self.pca.explained_variance_ratio_
        
        # 3D PCA
        X_pca_3d = self.pca_3d.fit_transform(X)
        variance_3d = self.pca_3d.explained_variance_ratio_
        
        print(f"2D PCA - PC1: {variance_2d[0]:.2%}, PC2: {variance_2d[1]:.2%}")
        print(f"Total variance preserved (2D): {sum(variance_2d):.2%}")
        print(f"Total variance preserved (3D): {sum(variance_3d):.2%}")
        
        return X_pca_2d, X_pca_3d, variance_2d, variance_3d
    
    def cluster_and_visualize(self, df):
        """Complete clustering pipeline with visualization data."""
        # Preprocess
        X_scaled, feature_cols = self.preprocess_data(df)
        
        # Cluster with both algorithms
        kmeans_labels = self.fit_kmeans(X_scaled)
        gmm_labels, gmm_probs = self.fit_gmm(X_scaled)
        
        # Apply PCA
        X_pca_2d, X_pca_3d, variance_2d, variance_3d = self.apply_pca(X_scaled)
        
        # Prepare results DataFrame
        results_df = df.copy()
        results_df['kmeans_cluster'] = kmeans_labels
        results_df['gmm_cluster'] = gmm_labels
        results_df['pca_1'] = X_pca_2d[:, 0]
        results_df['pca_2'] = X_pca_2d[:, 1]
        results_df['pca_3d_1'] = X_pca_3d[:, 0]
        results_df['pca_3d_2'] = X_pca_3d[:, 1]
        results_df['pca_3d_3'] = X_pca_3d[:, 2]
        
        # Add GMM probabilities
        for i in range(self.n_clusters):
            results_df[f'gmm_prob_{i}'] = gmm_probs[:, i]
        
        return results_df, X_scaled, variance_2d, variance_3d
    
    def get_cluster_statistics(self, df, labels, feature_cols):
        """Calculate statistics for each cluster."""
        cluster_stats = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            stats = {
                'cluster_id': int(cluster_id),
                'song_count': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(df) * 100),
                'filenames': cluster_data['filename'].tolist(),
                'genres': cluster_data['genre'].tolist(),
                'mean_features': {
                    col: float(cluster_data[col].mean())
                    for col in feature_cols
                },
                'std_features': {
                    col: float(cluster_data[col].std())
                    for col in feature_cols
                }
            }
            
            cluster_stats.append(stats)
        
        return cluster_stats
    
    def save_models(self, output_dir='models'):
        """Save trained models and transformers."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.kmeans, f'{output_dir}/kmeans_model.pkl')
        joblib.dump(self.gmm, f'{output_dir}/gmm_model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.pca, f'{output_dir}/pca_model.pkl')
        joblib.dump(self.pca_3d, f'{output_dir}/pca_3d_model.pkl')
        
        print(f"Models saved to {output_dir}/")
    
    def load_models(self, output_dir='models'):
        """Load pre-trained models."""
        self.kmeans = joblib.load(f'{output_dir}/kmeans_model.pkl')
        self.gmm = joblib.load(f'{output_dir}/gmm_model.pkl')
        self.scaler = joblib.load(f'{output_dir}/scaler.pkl')
        self.pca = joblib.load(f'{output_dir}/pca_model.pkl')
        self.pca_3d = joblib.load(f'{output_dir}/pca_3d_model.pkl')
        print("Models loaded successfully")


# Usage example
if __name__ == "__main__":
    # Load features
    df = pd.read_csv('data/processed/features_selected.csv')
    
    # Initialize clusterer
    clusterer = MusicClusterer(n_clusters=10)
    
    # Run clustering pipeline
    results_df, X_scaled, variance_2d, variance_3d = clusterer.cluster_and_visualize(df)
    
    # Get cluster statistics
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    cluster_stats = clusterer.get_cluster_statistics(
        results_df, 
        results_df['kmeans_cluster'].values,
        feature_cols
    )
    
    # Save results
    results_df.to_csv('data/processed/cluster_assignments.csv', index=False)
    
    import json
    with open('data/processed/cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    
    # Save models
    clusterer.save_models()
    
    print(f"Clustering complete! {len(results_df)} songs grouped into {len(cluster_stats)} clusters")
>>>>>>> c21d536 (Initial commit)
