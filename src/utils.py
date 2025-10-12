import numpy as np
import pandas as pd

def print_cluster_distribution(labels):
    """Print the distribution of songs across clusters"""
    unique, counts = np.unique(labels, return_counts=True)
    
    print("\nCluster Distribution:")
    print("-" * 30)
    for cluster, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"Cluster {cluster}: {count} songs ({percentage:.1f}%)")
    print("-" * 30)

def get_cluster_stats(df, labels, feature_columns):
    """
    Calculate statistics for each cluster
    
    Args:
        df: Dataframe with features
        labels: Cluster labels
        feature_columns: List of feature column names
    
    Returns:
        stats_df: Dataframe with cluster statistics
    """
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    stats = []
    for cluster in range(max(labels) + 1):
        cluster_data = df_copy[df_copy['cluster'] == cluster]
        
        cluster_stats = {
            'cluster': cluster,
            'count': len(cluster_data)
        }
        
        # Calculate mean for each feature
        for col in feature_columns[:5]:  # Top 5 features
            if col in df_copy.columns:
                cluster_stats[f'mean_{col}'] = cluster_data[col].mean()
        
        stats.append(cluster_stats)
    
    return pd.DataFrame(stats)
