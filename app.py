from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Color palette
def get_cluster_color(cluster_id):
    colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
        '#FF9F40', '#E7E9ED', '#8B5CF6', '#10B981', '#F59E0B'
    ]
    return colors[cluster_id % len(colors)]

app.jinja_env.globals.update(get_cluster_color=get_cluster_color)

# Check models exist
if not os.path.exists('models/kmeans_model.pkl'):
    print("ERROR: Models not found! Run: python train_models.py")
    exit(1)

# Load models
print("Loading models and data...")
kmeans_model = joblib.load('models/kmeans_model.pkl')
gmm_model = joblib.load('models/gmm_model.pkl')
pca_model = joblib.load('models/pca_model.pkl')

# Load metrics
with open('models/metrics.json', 'r') as f:
    metrics_data = json.load(f)

# Load data
df = pd.read_csv('data/processed/cleaned_features.csv')

# Load feature columns
with open('models/feature_columns.txt', 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print(f"✓ Loaded {len(df)} songs with {len(feature_columns)} features")

# Precompute predictions
X_scaled = df[feature_columns].values
X_pca = pca_model.transform(X_scaled)
kmeans_labels = kmeans_model.predict(X_scaled)
gmm_labels = gmm_model.predict(X_scaled)
gmm_probabilities = gmm_model.predict_proba(X_scaled)

@app.route('/')
def index():
    return render_template('index.html',
                         total_songs=metrics_data['dataset']['total_songs'],
                         n_clusters_kmeans=metrics_data['kmeans']['n_clusters'],
                         n_clusters_gmm=metrics_data['gmm']['n_components'],
                         explained_var=round(metrics_data['pca']['total_variance_explained'], 2))

@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with cluster details and song lists"""
    # Prepare comprehensive cluster data
    clusters_info = []
    
    for i in range(kmeans_model.n_clusters):
        mask = kmeans_labels == i
        cluster_songs = df[mask]
        
        # Get all songs in this cluster
        song_list = []
        if 'filename' in df.columns:
            song_list = cluster_songs['filename'].tolist()
        
        # Get genre distribution
        genre_dist = {}
        if 'label' in df.columns:
            genre_counts = cluster_songs['label'].value_counts()
            genre_dist = {str(k): int(v) for k, v in genre_counts.to_dict().items()}
            dominant_genre = genre_counts.index[0] if len(genre_counts) > 0 else 'Unknown'
        else:
            dominant_genre = 'Unknown'
        
        # Feature statistics
        features = {}
        for feat in ['tempo', 'energy', 'loudness', 'valence', 'danceability']:
            if feat in df.columns:
                features[feat] = {
                    'mean': float(cluster_songs[feat].mean()),
                    'std': float(cluster_songs[feat].std()),
                    'min': float(cluster_songs[feat].min()),
                    'max': float(cluster_songs[feat].max())
                }
        
        # Get PCA coordinates for this cluster
        cluster_pca_x = [X_pca[idx, 0] for idx, label in enumerate(kmeans_labels) if label == i]
        cluster_pca_y = [X_pca[idx, 1] for idx, label in enumerate(kmeans_labels) if label == i]
        
        clusters_info.append({
            'id': i,
            'size': int(mask.sum()),
            'percentage': round((mask.sum() / len(df)) * 100, 1),
            'songs': song_list,
            'genres': genre_dist,
            'dominant_genre': dominant_genre,
            'features': features,
            'pca_x': cluster_pca_x[:10],  # First 10 for preview
            'pca_y': cluster_pca_y[:10]
        })
    
    return render_template('dashboard.html', 
                         clusters=clusters_info,
                         total_songs=len(df),
                         total_clusters=kmeans_model.n_clusters,
                         metrics=metrics_data)


@app.route('/analysis')
def analysis():
    cluster_stats = []
    
    for i in range(kmeans_model.n_clusters):
        mask = kmeans_labels == i
        cluster_songs = df[mask]
        
        stats = {
            'cluster_id': i,
            'song_count': int(mask.sum()),
        }
        
        # Feature statistics
        display_features = [
            ('tempo', 'Tempo'),
            ('spectral_centroid_mean', 'Brightness'),
            ('spectral_bandwidth_mean', 'Bandwidth'),
            ('rolloff_mean', 'Rolloff'),
            ('zero_crossing_rate_mean', 'ZCR'),
            ('rms_mean', 'Energy'),
            ('mfcc1_mean', 'MFCC-1'),
        ]
        
        feature_stats = []
        for feature_col, display_name in display_features:
            if feature_col in df.columns:
                mean_val = float(cluster_songs[feature_col].mean())
                feature_stats.append({
                    'name': display_name,
                    'value': f"{mean_val:.3f}",
                    'raw_value': mean_val
                })
            if len(feature_stats) >= 5:
                break
        
        if len(feature_stats) < 3:
            for col in feature_columns[:5]:
                if col in df.columns:
                    mean_val = float(cluster_songs[col].mean())
                    feature_name = col.replace('_mean', '').replace('_', ' ').title()
                    feature_stats.append({
                        'name': feature_name,
                        'value': f"{mean_val:.3f}",
                        'raw_value': mean_val
                    })
        
        stats['features'] = feature_stats
        cluster_stats.append(stats)
    
    return render_template('analysis.html', cluster_stats=cluster_stats)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/metrics')
def metrics_page():
    """New page showing model performance"""
    return render_template('metrices.html', metrics=metrics_data)

@app.route('/api/clusters/kmeans')
def get_kmeans_clusters():
    clusters = {
        'labels': kmeans_labels.tolist(),
        'n_clusters': int(kmeans_model.n_clusters),
        'inertia': float(kmeans_model.inertia_)
    }
    return jsonify(clusters)

@app.route('/api/clusters/gmm')
def get_gmm_clusters():
    clusters = {
        'labels': gmm_labels.tolist(),
        'probabilities': gmm_probabilities.tolist(),
        'n_components': int(gmm_model.n_components)
    }
    return jsonify(clusters)

@app.route('/api/visualization/pca')
def get_pca_data():
    data = {
        'x': X_pca[:, 0].tolist(),
        'y': X_pca[:, 1].tolist(),
        'kmeans_labels': kmeans_labels.tolist(),
        'gmm_labels': gmm_labels.tolist(),
        'filenames': df['filename'].tolist() if 'filename' in df.columns else [],
        'genres': df['label'].tolist() if 'label' in df.columns else []
    }
    return jsonify(data)

@app.route('/cluster-report')
def cluster_report():
    """Comprehensive cluster analysis report"""
    # Calculate detailed cluster statistics
    cluster_details = []
    
    for i in range(kmeans_model.n_clusters):
        mask = kmeans_labels == i
        cluster_songs = df[mask]
        
        # Feature statistics
        feature_stats = {}
        for feat in ['tempo', 'energy', 'loudness', 'valence', 'danceability']:
            if feat in df.columns:
                feature_stats[feat] = {
                    'mean': float(cluster_songs[feat].mean()),
                    'std': float(cluster_songs[feat].std()),
                    'min': float(cluster_songs[feat].min()),
                    'max': float(cluster_songs[feat].max())
                }
        
        # Genre distribution if available
        genre_dist = {}
        if 'label' in df.columns:
            genre_counts = cluster_songs['label'].value_counts()
            genre_dist = {str(k): int(v) for k, v in genre_counts.items()}
        
        # Top songs
        top_songs = []
        if 'filename' in df.columns:
            top_songs = cluster_songs['filename'].head(5).tolist()
        
        # Cluster characteristics
        cluster_details.append({
            'id': i,
            'size': int(mask.sum()),
            'percentage': round((mask.sum() / len(df)) * 100, 1),
            'features': feature_stats,
            'genre_distribution': genre_dist,
            'top_songs': top_songs,
            'dominant_genre': max(genre_dist.items(), key=lambda x: x[1])[0] if genre_dist else 'Unknown'
        })
    
    return render_template('cluster_report.html', 
                         clusters=cluster_details,
                         total_songs=len(df),
                         metrics=metrics_data)

@app.route('/api/cluster/<int:cluster_id>/details')
def get_cluster_full_details(cluster_id):
    """API endpoint for detailed cluster information"""
    if cluster_id < 0 or cluster_id >= kmeans_model.n_clusters:
        return jsonify({'error': 'Invalid cluster ID'}), 400
    
    mask = kmeans_labels == cluster_id
    cluster_data = df[mask]
    
    # Detailed feature analysis
    features_analysis = {}
    for feat in ['tempo', 'energy', 'loudness', 'valence', 'danceability']:
        if feat in df.columns:
            features_analysis[feat] = {
                'mean': float(cluster_data[feat].mean()),
                'median': float(cluster_data[feat].median()),
                'std': float(cluster_data[feat].std()),
                'min': float(cluster_data[feat].min()),
                'max': float(cluster_data[feat].max()),
                'values': cluster_data[feat].tolist()
            }
    
    # Genre breakdown
    genre_breakdown = {}
    if 'label' in df.columns:
        genre_counts = cluster_data['label'].value_counts()
        genre_breakdown = {str(k): int(v) for k, v in genre_counts.to_dict().items()}
    
    return jsonify({
        'cluster_id': cluster_id,
        'size': int(mask.sum()),
        'features': features_analysis,
        'genres': genre_breakdown,
        'songs': cluster_data['filename'].tolist() if 'filename' in df.columns else []
    })


@app.route('/api/metrics')
def get_metrics():
    """Return all evaluation metrics"""
    return jsonify(metrics_data)

@app.route('/api/cluster/distribution')
def get_cluster_distribution():
    """Return cluster size distribution"""
    kmeans_dist = {}
    for label in kmeans_labels:
        kmeans_dist[int(label)] = kmeans_dist.get(int(label), 0) + 1
    
    gmm_dist = {}
    for label in gmm_labels:
        gmm_dist[int(label)] = gmm_dist.get(int(label), 0) + 1
    
    return jsonify({
        'kmeans': kmeans_dist,
        'gmm': gmm_dist
    })

@app.route('/api/cluster/<int:cluster_id>/features')
def get_cluster_features(cluster_id):
    """Get average features for a specific cluster"""
    if cluster_id < 0 or cluster_id >= kmeans_model.n_clusters:
        return jsonify({'error': 'Invalid cluster ID'}), 400
    
    mask = kmeans_labels == cluster_id
    cluster_data = df[mask]
    
    # Calculate averages for top features
    features = {}
    for col in feature_columns[:8]:
        if col in df.columns:
            features[col] = float(cluster_data[col].mean())
    
    return jsonify({
        'cluster_id': cluster_id,
        'song_count': int(mask.sum()),
        'features': features
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 Music Genre Clustering Dashboard")
    print("="*60)
    print("Dashboard: http://localhost:5000")
    print("Metrics: http://localhost:5000/metrics")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
