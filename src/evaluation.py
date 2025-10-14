import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

class ClusteringEvaluator:
    """Evaluate clustering performance with multiple metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_all(self, X, labels, model_name='K-Means'):
        """Calculate all evaluation metrics."""
        print(f"\nEvaluating {model_name} clustering...")
        
        # Silhouette Score
        silhouette = silhouette_score(X, labels)
        self.metrics[f'{model_name}_silhouette'] = silhouette
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(X, labels)
        self.metrics[f'{model_name}_davies_bouldin'] = davies_bouldin
        
        # Calinski-Harabasz Index
        calinski = calinski_harabasz_score(X, labels)
        self.metrics[f'{model_name}_calinski_harabasz'] = calinski
        
        return {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'calinski_harabasz_index': float(calinski)
        }
    
    def interpret_metrics(self, metrics):
        """Provide interpretation of metric values."""
        interpretations = {}
        
        # Silhouette Score interpretation
        sil = metrics['silhouette_score']
        if sil < 0.20:
            sil_rating = "Weak structure"
            sil_stars = "⭐"
        elif sil < 0.40:
            sil_rating = "Good structure"
            sil_stars = "⭐⭐⭐"
        elif sil < 0.70:
            sil_rating = "Strong structure"
            sil_stars = "⭐⭐⭐⭐"
        else:
            sil_rating = "Excellent structure"
            sil_stars = "⭐⭐⭐⭐⭐"
        
        interpretations['silhouette'] = {
            'value': sil,
            'rating': sil_rating,
            'stars': sil_stars
        }
        
        # Davies-Bouldin Index interpretation
        db = metrics['davies_bouldin_index']
        if db < 1.0:
            db_rating = "Excellent separation"
            db_stars = "⭐⭐⭐⭐⭐"
        elif db < 1.5:
            db_rating = "Good separation"
            db_stars = "⭐⭐⭐⭐"
        elif db < 2.0:
            db_rating = "Fair separation"
            db_stars = "⭐⭐⭐"
        else:
            db_rating = "Poor separation"
            db_stars = "⭐⭐"
        
        interpretations['davies_bouldin'] = {
            'value': db,
            'rating': db_rating,
            'stars': db_stars
        }
        
        # Calinski-Harabasz Index interpretation
        ch = metrics['calinski_harabasz_index']
        if ch < 100:
            ch_rating = "Weak clustering"
            ch_stars = "⭐⭐"
        elif ch < 200:
            ch_rating = "Fair clustering"
            ch_stars = "⭐⭐⭐"
        elif ch < 400:
            ch_rating = "Good clustering"
            ch_stars = "⭐⭐⭐⭐"
        else:
            ch_rating = "Very good clustering"
            ch_stars = "⭐⭐⭐⭐⭐"
        
        interpretations['calinski_harabasz'] = {
            'value': ch,
            'rating': ch_rating,
            'stars': ch_stars
        }
        
        return interpretations
    
    def compare_algorithms(self, X, kmeans_labels, gmm_labels):
        """Compare K-Means and GMM performance."""
        print("\n" + "="*50)
        print("ALGORITHM COMPARISON")
        print("="*50)
        
        kmeans_metrics = self.evaluate_all(X, kmeans_labels, 'K-Means')
        gmm_metrics = self.evaluate_all(X, gmm_labels, 'GMM')
        
        # Print comparison
        print(f"\n{'Metric':<30} {'K-Means':<15} {'GMM':<15} {'Winner'}")
        print("-" * 70)
        
        # Silhouette Score (higher is better)
        print(f"{'Silhouette Score':<30} {kmeans_metrics['silhouette_score']:<15.4f} "
              f"{gmm_metrics['silhouette_score']:<15.4f} "
              f"{'K-Means' if kmeans_metrics['silhouette_score'] > gmm_metrics['silhouette_score'] else 'GMM'}")
        
        # Davies-Bouldin (lower is better)
        print(f"{'Davies-Bouldin Index':<30} {kmeans_metrics['davies_bouldin_index']:<15.4f} "
              f"{gmm_metrics['davies_bouldin_index']:<15.4f} "
              f"{'K-Means' if kmeans_metrics['davies_bouldin_index'] < gmm_metrics['davies_bouldin_index'] else 'GMM'}")
        
        # Calinski-Harabasz (higher is better)
        print(f"{'Calinski-Harabasz Index':<30} {kmeans_metrics['calinski_harabasz_index']:<15.4f} "
              f"{gmm_metrics['calinski_harabasz_index']:<15.4f} "
              f"{'K-Means' if kmeans_metrics['calinski_harabasz_index'] > gmm_metrics['calinski_harabasz_index'] else 'GMM'}")
        
        return {
            'kmeans': self.interpret_metrics(kmeans_metrics),
            'gmm': self.interpret_metrics(gmm_metrics)
        }


# Usage example
if __name__ == "__main__":
    import pandas as pd
    import joblib
    
    # Load data and models
    df = pd.read_csv('data/processed/cluster_assignments.csv')
    scaler = joblib.load('models/scaler.pkl')
    
    # Prepare data
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Get labels
    kmeans_labels = df['kmeans_cluster'].values
    gmm_labels = df['gmm_cluster'].values
    
    # Evaluate
    evaluator = ClusteringEvaluator()
    comparison = evaluator.compare_algorithms(X_scaled, kmeans_labels, gmm_labels)
    
    # Save results
    import json
    with open('data/processed/evaluation_metrics.json', 'w') as f:
        json.dump(comparison, f, indent=2)
