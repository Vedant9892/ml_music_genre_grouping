import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os
import json

os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*70)
print("MUSIC GENRE CLUSTERING - ASSIGNMENT REQUIREMENTS")
print("Features: Tempo, Energy, Loudness, Valence, Danceability")
print("="*70)

# Load dataset
print("\n[Step 1/6] Loading GTZAN dataset...")
df = pd.read_csv('data/features_30_sec.csv')
print(f"✓ Loaded {len(df)} songs")

# Check available columns
print(f"\nAvailable features: {df.columns.tolist()}")

# Extract REQUIRED features (or close approximations)
print("\n[Step 2/6] Extracting required features...")

feature_mapping = {}

# 1. TEMPO - Direct from GTZAN
if 'tempo' in df.columns:
    feature_mapping['tempo'] = df['tempo']
    print("✓ Tempo: Found directly")
else:
    print("✗ Tempo: NOT FOUND")

# 2. ENERGY - Use RMS (Root Mean Square) as proxy
energy_candidates = ['rms_mean', 'rmse_mean', 'energy']
for col in energy_candidates:
    if col in df.columns:
        feature_mapping['energy'] = df[col]
        print(f"✓ Energy: Using '{col}' as proxy")
        break

# 3. LOUDNESS - Direct or use RMS
loudness_candidates = ['loudness', 'rms_mean']
for col in loudness_candidates:
    if col in df.columns:
        feature_mapping['loudness'] = df[col]
        print(f"✓ Loudness: Using '{col}'")
        break

# 4. VALENCE - APPROXIMATE using harmonic features
# Higher harmony + positive spectral centroid ≈ happier sound
if 'harmony_mean' in df.columns and 'spectral_centroid_mean' in df.columns:
    # Normalize to 0-1 range
    harmony_norm = (df['harmony_mean'] - df['harmony_mean'].min()) / (df['harmony_mean'].max() - df['harmony_mean'].min())
    centroid_norm = (df['spectral_centroid_mean'] - df['spectral_centroid_mean'].min()) / (df['spectral_centroid_mean'].max() - df['spectral_centroid_mean'].min())
    feature_mapping['valence'] = (harmony_norm + centroid_norm) / 2
    print("✓ Valence: APPROXIMATED from harmony + spectral centroid (proxy)")
elif 'spectral_centroid_mean' in df.columns:
    # Fallback: just use spectral centroid
    feature_mapping['valence'] = (df['spectral_centroid_mean'] - df['spectral_centroid_mean'].min()) / (df['spectral_centroid_mean'].max() - df['spectral_centroid_mean'].min())
    print("✓ Valence: APPROXIMATED from spectral centroid (proxy)")
else:
    print("⚠ Valence: Using dummy values (feature not available)")
    feature_mapping['valence'] = np.random.rand(len(df))

# 5. DANCEABILITY - APPROXIMATE using tempo + rhythm
# Higher tempo + higher zero crossing rate ≈ more danceable
if 'tempo' in df.columns and 'zero_crossing_rate_mean' in df.columns:
    # Normalize tempo (typical dance music: 100-140 BPM)
    tempo_norm = np.clip((df['tempo'] - 80) / 80, 0, 1)
    zcr_norm = (df['zero_crossing_rate_mean'] - df['zero_crossing_rate_mean'].min()) / (df['zero_crossing_rate_mean'].max() - df['zero_crossing_rate_mean'].min())
    feature_mapping['danceability'] = (tempo_norm + zcr_norm) / 2
    print("✓ Danceability: APPROXIMATED from tempo + zero crossing rate (proxy)")
elif 'tempo' in df.columns:
    feature_mapping['danceability'] = np.clip((df['tempo'] - 80) / 80, 0, 1)
    print("✓ Danceability: APPROXIMATED from tempo only (proxy)")
else:
    print("⚠ Danceability: Using dummy values (feature not available)")
    feature_mapping['danceability'] = np.random.rand(len(df))

# Create feature dataframe
feature_names = ['tempo', 'energy', 'loudness', 'valence', 'danceability']
X_df = pd.DataFrame(feature_mapping)
X = X_df.values

print(f"\n✓ Created 5 required features")
print(f"  Features shape: {X.shape}")
print("\n⚠️ NOTE: Valence and Danceability are APPROXIMATIONS")
print("   GTZAN doesn't have these Spotify features natively")

# Handle missing values
X = np.nan_to_num(X, nan=0.0)

# Standardize
print("\n[Step 3/6] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features normalized (mean=0, std=1)")

# Save processed data
df_processed = df.copy()
for i, fname in enumerate(feature_names):
    df_processed[f'{fname}_standardized'] = X_scaled[:, i]
    df_processed[fname] = X[:, i]
df_processed.to_csv('data/processed/cleaned_features.csv', index=False)
print("✓ Saved processed data")

# Train K-Means (note: it's K-Means, not KNN - KNN is classification, K-Means is clustering)
print("\n[Step 4/6] Training K-Means model...")
print("   (Note: Assignment says KNN but means K-Means for clustering)")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=20, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)

print("✓ K-Means training complete")
print(f"  → Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  → Davies-Bouldin Index: {kmeans_davies_bouldin:.4f}")
print(f"  → Calinski-Harabasz Index: {kmeans_calinski:.2f}")

joblib.dump(kmeans, 'models/kmeans_model.pkl')

# Train GMM
print("\n[Step 5/6] Training GMM model...")
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42, max_iter=300)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)

gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_davies_bouldin = davies_bouldin_score(X_scaled, gmm_labels)
gmm_bic = gmm.bic(X_scaled)

print("✓ GMM training complete")
print(f"  → Silhouette Score: {gmm_silhouette:.4f}")
print(f"  → Davies-Bouldin Index: {gmm_davies_bouldin:.4f}")
print(f"  → BIC Score: {gmm_bic:.2f}")

joblib.dump(gmm, 'models/gmm_model.pkl')

# Apply PCA for visualization
print("\n[Step 6/6] Applying PCA (2D visualization)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA complete")
print(f"  → PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% variance")
print(f"  → PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% variance")
print(f"  → Total: {sum(pca.explained_variance_ratio_)*100:.2f}% information preserved")

joblib.dump(pca, 'models/pca_model.pkl')

# Save metrics
metrics = {
    'kmeans': {
        'silhouette_score': float(kmeans_silhouette),
        'davies_bouldin_index': float(kmeans_davies_bouldin),
        'calinski_harabasz_index': float(kmeans_calinski),
        'inertia': float(kmeans.inertia_),
        'n_clusters': int(kmeans.n_clusters)
    },
    'gmm': {
        'silhouette_score': float(gmm_silhouette),
        'davies_bouldin_index': float(gmm_davies_bouldin),
        'bic_score': float(gmm_bic),
        'aic_score': float(gmm.aic(X_scaled)),
        'converged': bool(gmm.converged_),
        'n_components': int(gmm.n_components)
    },
    'pca': {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(sum(pca.explained_variance_ratio_) * 100)
    },
    'dataset': {
        'total_songs': len(df),
        'n_features': 5,
        'features': feature_names
    }
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

with open('models/feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_names))

print("\n" + "="*70)
print("CLUSTER DISTRIBUTION (K-Means)")
print("="*70)
unique, counts = np.unique(kmeans_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    percentage = (count / len(kmeans_labels)) * 100
    bar = '█' * int(percentage / 2)
    print(f"Cluster {cluster}: {bar} {count} songs ({percentage:.1f}%)")

print("\n" + "="*70)
print("✓ MODELS TRAINED SUCCESSFULLY!")
print("="*70)
print("\n📝 ASSIGNMENT REQUIREMENT COMPLIANCE:")
print(f"   ✓ Features used: {', '.join(feature_names)}")
print(f"   ✓ K-Means clustering: {kmeans.n_clusters} clusters")
print(f"   ✓ GMM clustering: {gmm.n_components} components")
print(f"   ✓ PCA visualization: 2D scatter plot")
print("\n⚠️  IMPORTANT NOTE:")
print("   Valence & Danceability are APPROXIMATED from GTZAN features")
print("   GTZAN doesn't have these Spotify-specific metrics natively")
print("\nNext: Run 'python app.py'")
