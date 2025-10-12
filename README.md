# Music Genre Clustering Project

ML project that groups similar songs based on audio features using unsupervised learning.

## Features
- 🎯 K-Means clustering (Silhouette Score: 0.2434)
- 🎲 GMM clustering (probabilistic assignments)
- 📊 PCA visualization (90.51% variance in 2D, 99.81% in 3D)
- 🌙 Modern dark theme dashboard
- 🎨 Interactive visualizations (2D/3D toggle)
- 📈 Horizontal bar charts for feature analysis
- 🎵 All 5 audio features: Tempo, Energy, Loudness, Valence, Danceability

## Dataset
GTZAN Dataset with 1,000 songs across 10 genres

## Technologies
- **Backend:** Python, Flask, scikit-learn
- **ML:** K-Means, GMM, PCA
- **Audio:** librosa
- **Frontend:** HTML, CSS (Dark Theme), JavaScript
- **Visualization:** Plotly, Chart.js

## Installation

\`\`\`bash
# Clone repository
git clone https://github.com/Vedant9892/ml_music_genre_grouping.git
cd ml_music_genre_grouping

# Install dependencies
pip install -r requirements.txt

# Run feature extraction
python main.py

# Launch dashboard
python run.py
\`\`\`

## Usage
Visit `http://localhost:5000` after running the dashboard.

## Screenshots
(Add screenshots of your dark theme dashboard here)

## Performance
- **K-Means Silhouette Score:** 0.2434 ⭐⭐⭐
- **Davies-Bouldin Index:** 1.1256 ⭐⭐⭐⭐
- **Calinski-Harabasz Index:** 401.22 ⭐⭐⭐⭐⭐
- **PCA Variance (2D):** 90.51% ⭐⭐⭐⭐⭐

## Author
**Vedant** - [GitHub Profile](https://github.com/Vedant9892)
