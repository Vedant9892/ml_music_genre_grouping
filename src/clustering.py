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
