"""
Complete pipeline for Music Genre Clustering Project
Runs feature extraction, clustering, evaluation, and launches dashboard
"""

import pandas as pd
from pathlib import Path
from src.feature_extraction import AudioFeatureExtractor
from src.clustering import MusicClusterer
from src.evaluation import ClusteringEvaluator
import json

def run_complete_pipeline():
    """Execute the complete ML pipeline."""
    
    print("=" * 70)
    print("MUSIC GENRE CLUSTERING PROJECT - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Feature Extraction
    print("\n[Step 1/5] Extracting audio features from GTZAN dataset...")
    print("-" * 70)
    
    extractor = AudioFeatureExtractor('data/gtzan/genres_orignal')
    features_df = extractor.save_features('data/processed/features_selected.csv')
    
    print(f"✅ Extracted {len(features_df)} songs with 5 features")
    print(f"   Features: tempo, energy, loudness, valence, danceability")
    
    # Step 2: Clustering
    print("\n[Step 2/5] Training clustering models...")
    print("-" * 70)
    
    clusterer = MusicClusterer(n_clusters=10)
    results_df, X_scaled, variance_2d, variance_3d = clusterer.cluster_and_visualize(features_df)
    
    print(f"✅ K-Means clustering complete")
    print(f"✅ GMM clustering complete")
    print(f"✅ PCA applied - 2D variance: {sum(variance_2d):.2%}, 3D variance: {sum(variance_3d):.2%}")
    
    # Step 3: Get Cluster Statistics
    print("\n[Step 3/5] Calculating cluster statistics...")
    print("-" * 70)
    
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    cluster_stats = clusterer.get_cluster_statistics(
        results_df,
        results_df['kmeans_cluster'].values,
        feature_cols
    )
    
    print(f"✅ Generated statistics for {len(cluster_stats)} clusters")
    for stat in cluster_stats:
        print(f"   Cluster {stat['cluster_id']}: {stat['song_count']} songs ({stat['percentage']:.1f}%)")
    
    # Step 4: Evaluation
    print("\n[Step 4/5] Evaluating clustering performance...")
    print("-" * 70)
    
    evaluator = ClusteringEvaluator()
    comparison = evaluator.compare_algorithms(
        X_scaled,
        results_df['kmeans_cluster'].values,
        results_df['gmm_cluster'].values
    )
    
    print(f"\n✅ Evaluation complete")
    print(f"   K-Means Silhouette Score: {comparison['kmeans']['silhouette']['value']:.4f} {comparison['kmeans']['silhouette']['stars']}")
    print(f"   K-Means Davies-Bouldin: {comparison['kmeans']['davies_bouldin']['value']:.4f} {comparison['kmeans']['davies_bouldin']['stars']}")
    print(f"   K-Means Calinski-Harabasz: {comparison['kmeans']['calinski_harabasz']['value']:.2f} {comparison['kmeans']['calinski_harabasz']['stars']}")
    
    # Step 5: Save all results
    print("\n[Step 5/5] Saving results...")
    print("-" * 70)
    
    results_df.to_csv('data/processed/cluster_assignments.csv', index=False)
    
    with open('data/processed/cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    
    with open('data/processed/evaluation_metrics.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    clusterer.save_models()
    
    print("✅ All results saved successfully")
    print("   - data/processed/cluster_assignments.csv")
    print("   - data/processed/cluster_statistics.json")
    print("   - data/processed/evaluation_metrics.json")
    print("   - models/ (all model files)")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE! 🎉")
    print("=" * 70)
    print("\nProject Summary:")
    print(f"  📊 Dataset: GTZAN (1,000 songs, 10 genres)")
    print(f"  🎯 Features: 5 audio characteristics")
    print(f"  🔮 Clusters: 10 discovered clusters")
    print(f"  ⭐ Performance: Silhouette={comparison['kmeans']['silhouette']['value']:.4f}, DB={comparison['kmeans']['davies_bouldin']['value']:.4f}")
    print(f"  📈 PCA Variance: {sum(variance_2d):.2%} (2D), {sum(variance_3d):.2%} (3D)")
    print("\n🚀 Ready to launch dashboard!")
    print("   Run: python run.py")
    print("   Then visit: http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    # Create necessary directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    run_complete_pipeline()
