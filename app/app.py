from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load data and models
DATA_DIR = Path('data/processed')
MODELS_DIR = Path('models')

# Load cluster data
cluster_df = pd.read_csv(DATA_DIR / 'cluster_assignments.csv')
with open(DATA_DIR / 'cluster_statistics.json', 'r') as f:
    cluster_stats = json.load(f)

with open(DATA_DIR / 'evaluation_metrics.json', 'r') as f:
    evaluation_metrics = json.load(f)

# Load models
kmeans_model = joblib.load(MODELS_DIR / 'kmeans_model.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
pca_model = joblib.load(MODELS_DIR / 'pca_model.pkl')


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/model_info')
def get_model_info():
    """Return model features and performance metrics."""
    
    # Feature information - ALL 5 FEATURES
    features = [
        {
            'name': 'Tempo',
            'unit': 'BPM',
            'description': 'Speed of music (beats per minute)',
            'extraction': 'Direct extraction using beat tracking'
        },
        {
            'name': 'Energy',
            'unit': 'RMS',
            'description': 'Intensity and loudness level',
            'extraction': 'Root Mean Square (RMS) of amplitude'
        },
        {
            'name': 'Loudness',
            'unit': 'RMS',
            'description': 'Overall amplitude',
            'extraction': 'RMS mean across audio signal'
        },
        {
            'name': 'Valence',
            'unit': 'Normalized',
            'description': 'Musical positiveness (0-1)',
            'extraction': 'Approximated from harmony + spectral centroid'
        },
        {
            'name': 'Danceability',
            'unit': 'Normalized',
            'description': 'Dance suitability (0-1)',
            'extraction': 'Approximated from tempo + zero crossing rate'
        }
    ]
    
    # Performance metrics from actual results
    performance = {
        'kmeans': {
            'silhouette_score': 0.2434,
            'silhouette_rating': 'Good Performance',
            'silhouette_stars': '⭐⭐⭐',
            'davies_bouldin_index': 1.1256,
            'davies_bouldin_rating': 'Excellent Separation',
            'davies_bouldin_stars': '⭐⭐⭐⭐',
            'calinski_harabasz_index': 401.22,
            'calinski_harabasz_rating': 'Very Good Clustering',
            'calinski_harabasz_stars': '⭐⭐⭐⭐⭐',
            'algorithm_type': 'Hard Clustering (Partitioning)',
            'parameters': {
                'n_clusters': 10,
                'n_init': 20,
                'max_iter': 300
            }
        },
        'gmm': {
            'silhouette_score': 0.1142,
            'silhouette_rating': 'Weak Structure',
            'silhouette_stars': '⭐',
            'davies_bouldin_index': 1.6074,
            'davies_bouldin_rating': 'Fair Separation',
            'davies_bouldin_stars': '⭐⭐⭐',
            'algorithm_type': 'Soft Clustering (Probabilistic)',
            'parameters': {
                'n_components': 10,
                'covariance_type': 'full',
                'max_iter': 300
            }
        },
        'pca': {
            'variance_2d': 90.51,
            'pc1_variance': 51.32,
            'pc2_variance': 39.19,
            'variance_3d': 99.81,
            'rating': 'Outstanding Performance',
            'stars': '⭐⭐⭐⭐⭐'
        }
    }
    
    return jsonify({
        'features': features,
        'performance': performance,
        'dataset': {
            'name': 'GTZAN Music Genre Dataset',
            'total_songs': 1000,
            'genres': 10,
            'songs_per_genre': 100,
            'duration_per_song': '30 seconds'
        }
    })


@app.route('/api/clusters')
def get_clusters():
    """Return cluster summary information."""
    return jsonify({
        'total_clusters': len(cluster_stats),
        'total_songs': len(cluster_df),
        'clusters': cluster_stats
    })


@app.route('/api/cluster/<int:cluster_id>')
def get_cluster_detail(cluster_id):
    """
    Return detailed information for a specific cluster.
    **UPDATED VERSION** - Now includes all 5 features for radar chart.
    """
    if cluster_id < 0 or cluster_id >= len(cluster_stats):
        return jsonify({'error': 'Invalid cluster ID'}), 400
    
    cluster_data = cluster_stats[cluster_id]
    
    # Get songs in this cluster
    cluster_songs = cluster_df[cluster_df['kmeans_cluster'] == cluster_id]
    
    # ALL 5 FEATURES - ensure they exist in DataFrame
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in cluster_songs.columns]
    if missing_cols:
        return jsonify({
            'error': f'Missing columns in dataset: {missing_cols}',
            'available_columns': list(cluster_songs.columns)
        }), 500
    
    # Extract features for distance calculation
    cluster_features = cluster_songs[feature_cols].values
    centroid = kmeans_model.cluster_centers_[cluster_id]
    
    # Calculate distances from centroid
    distances = np.linalg.norm(
        scaler.transform(cluster_features) - centroid,
        axis=1
    )
    
    # Prepare song details with ALL 5 features
    songs = []
    for idx, (_, row) in enumerate(cluster_songs.iterrows()):
        songs.append({
            'filename': row['filename'],
            'genre': row['genre'],
            'distance_from_centroid': float(distances[idx]),
            'features': {
                'tempo': float(row['tempo']),
                'energy': float(row['energy']),
                'loudness': float(row['loudness']),
                'valence': float(row['valence']),
                'danceability': float(row['danceability'])
            }
        })
    
    # Sort by distance (most similar first)
    songs.sort(key=lambda x: x['distance_from_centroid'])
    
    # Calculate mean features for radar chart - ALL 5 FEATURES
    mean_features = {
        'tempo': float(cluster_songs['tempo'].mean()),
        'energy': float(cluster_songs['energy'].mean()),
        'loudness': float(cluster_songs['loudness'].mean()),
        'valence': float(cluster_songs['valence'].mean()),
        'danceability': float(cluster_songs['danceability'].mean())
    }
    
    # Calculate standard deviations for reference
    std_features = {
        'tempo': float(cluster_songs['tempo'].std()),
        'energy': float(cluster_songs['energy'].std()),
        'loudness': float(cluster_songs['loudness'].std()),
        'valence': float(cluster_songs['valence'].std()),
        'danceability': float(cluster_songs['danceability'].std())
    }
    
    return jsonify({
        'cluster_id': cluster_id,
        'summary': cluster_data,
        'mean_features': mean_features,
        'std_features': std_features,
        'songs': songs,
        'most_similar_songs': songs[:10],  # Top 10 most representative
        'feature_count': len(mean_features),  # Should be 5
        'total_songs': len(songs)
    })


@app.route('/api/pca_data')
def get_pca_data():
    """Return PCA-transformed coordinates for visualization."""
    
    # Prepare 2D data
    pca_2d_data = []
    for _, row in cluster_df.iterrows():
        pca_2d_data.append({
            'filename': row['filename'],
            'genre': row['genre'],
            'cluster': int(row['kmeans_cluster']),
            'x': float(row['pca_1']),
            'y': float(row['pca_2'])
        })
    
    # Prepare 3D data
    pca_3d_data = []
    for _, row in cluster_df.iterrows():
        pca_3d_data.append({
            'filename': row['filename'],
            'genre': row['genre'],
            'cluster': int(row['kmeans_cluster']),
            'x': float(row['pca_3d_1']),
            'y': float(row['pca_3d_2']),
            'z': float(row['pca_3d_3'])
        })
    
    # Cluster centroids in PCA space
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    centroids_scaled = kmeans_model.cluster_centers_
    centroids_pca_2d = pca_model.transform(centroids_scaled)
    
    centroids_2d = [
        {
            'cluster': i,
            'x': float(centroids_pca_2d[i, 0]),
            'y': float(centroids_pca_2d[i, 1])
        }
        for i in range(len(centroids_pca_2d))
    ]
    
    return jsonify({
        'pca_2d': pca_2d_data,
        'pca_3d': pca_3d_data,
        'centroids_2d': centroids_2d,
        'variance_explained': {
            '2d': {
                'pc1': 51.32,
                'pc2': 39.19,
                'total': 90.51
            },
            '3d': {
                'pc1': 51.32,
                'pc2': 39.19,
                'pc3': 9.30,
                'total': 99.81
            }
        }
    })


@app.route('/api/search')
def search_songs():
    """Search for songs by filename or genre."""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify({'results': []})
    
    # Filter songs
    results = cluster_df[
        cluster_df['filename'].str.lower().str.contains(query) |
        cluster_df['genre'].str.lower().str.contains(query)
    ]
    
    search_results = []
    for _, row in results.iterrows():
        search_results.append({
            'filename': row['filename'],
            'genre': row['genre'],
            'cluster': int(row['kmeans_cluster'])
        })
    
    return jsonify({'results': search_results[:50]})  # Limit to 50 results


@app.route('/api/cluster/<int:cluster_id>/songs')
def get_cluster_songs_detailed(cluster_id):
    """
    Additional endpoint for getting ALL songs in a cluster with full feature details.
    """
    if cluster_id < 0 or cluster_id >= 10:
        return jsonify({'error': 'Invalid cluster ID'}), 400
    
    # Get all songs in cluster
    cluster_songs = cluster_df[cluster_df['kmeans_cluster'] == cluster_id]
    
    # Feature columns
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    
    # Prepare detailed song list
    songs_detailed = []
    for _, row in cluster_songs.iterrows():
        songs_detailed.append({
            'filename': row['filename'],
            'genre': row['genre'],
            'cluster': int(row['kmeans_cluster']),
            'features': {
                'tempo': float(row['tempo']),
                'energy': float(row['energy']),
                'loudness': float(row['loudness']),
                'valence': float(row['valence']),
                'danceability': float(row['danceability'])
            },
            'pca_coordinates': {
                '2d': {
                    'x': float(row['pca_1']),
                    'y': float(row['pca_2'])
                },
                '3d': {
                    'x': float(row['pca_3d_1']),
                    'y': float(row['pca_3d_2']),
                    'z': float(row['pca_3d_3'])
                }
            }
        })
    
    return jsonify({
        'cluster_id': cluster_id,
        'total_songs': len(songs_detailed),
        'songs': songs_detailed
    })


@app.route('/api/stats')
def get_overall_stats():
    """
    Return overall dataset and clustering statistics.
    """
    feature_cols = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
    
    overall_stats = {
        'dataset': {
            'total_songs': len(cluster_df),
            'total_clusters': 10,
            'features_used': feature_cols
        },
        'feature_ranges': {
            col: {
                'min': float(cluster_df[col].min()),
                'max': float(cluster_df[col].max()),
                'mean': float(cluster_df[col].mean()),
                'std': float(cluster_df[col].std())
            }
            for col in feature_cols
        },
        'cluster_sizes': {
            f'cluster_{i}': int((cluster_df['kmeans_cluster'] == i).sum())
            for i in range(10)
        }
    }
    
    return jsonify(overall_stats)


if __name__ == '__main__':
    print("=" * 60)
    print("🎵 MUSIC GENRE CLUSTERING DASHBOARD")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  GET  /                              - Main dashboard")
    print("  GET  /api/model_info                - Model features & metrics")
    print("  GET  /api/clusters                  - All clusters summary")
    print("  GET  /api/cluster/<id>              - Specific cluster details")
    print("  GET  /api/cluster/<id>/songs        - All songs in cluster")
    print("  GET  /api/pca_data                  - PCA visualization data")
    print("  GET  /api/search?q=<query>          - Search songs/genres")
    print("  GET  /api/stats                     - Overall statistics")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
