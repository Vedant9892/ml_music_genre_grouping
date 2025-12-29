// Dashboard main functionality
class MusicClusterDashboard {
    constructor() {
        this.currentView = '2d';
        this.selectedCluster = null;
        this.init();
    }

    async init() {
        await this.loadModelInfo();
        await this.loadClusters();
        await this.loadPCAData();
        this.setupEventListeners();
    }

    async loadModelInfo() {
        const response = await fetch('/api/model_info');
        const data = await response.json();
        this.renderModelInfo(data);
    }

    renderModelInfo(data) {
        // Render features
        const featuresContainer = document.getElementById('features-container');
        featuresContainer.innerHTML = data.features.map(feature => `
            <div class="feature-card">
                <h4>${feature.name}</h4>
                <p class="feature-unit">${feature.unit}</p>
                <p class="feature-desc">${feature.description}</p>
                <p class="feature-extraction"><strong>Extraction:</strong> ${feature.extraction}</p>
            </div>
        `).join('');

        // Render K-Means metrics
        const kmeansMetrics = data.performance.kmeans;
        document.getElementById('kmeans-metrics').innerHTML = `
            <div class="metric-card">
                <h4>Silhouette Score</h4>
                <div class="metric-value">${kmeansMetrics.silhouette_score.toFixed(4)}</div>
                <div class="metric-rating">${kmeansMetrics.silhouette_stars} ${kmeansMetrics.silhouette_rating}</div>
            </div>
            <div class="metric-card">
                <h4>Davies-Bouldin Index</h4>
                <div class="metric-value">${kmeansMetrics.davies_bouldin_index.toFixed(4)}</div>
                <div class="metric-rating">${kmeansMetrics.davies_bouldin_stars} ${kmeansMetrics.davies_bouldin_rating}</div>
            </div>
            <div class="metric-card">
                <h4>Calinski-Harabasz Index</h4>
                <div class="metric-value">${kmeansMetrics.calinski_harabasz_index.toFixed(2)}</div>
                <div class="metric-rating">${kmeansMetrics.calinski_harabasz_stars} ${kmeansMetrics.calinski_harabasz_rating}</div>
            </div>
            <div class="metric-card">
                <h4>PCA Variance Preserved</h4>
                <div class="metric-value">${data.performance.pca.variance_2d.toFixed(2)}%</div>
                <div class="metric-rating">${data.performance.pca.stars} ${data.performance.pca.rating}</div>
            </div>
        `;
    }

    async loadClusters() {
        const response = await fetch('/api/clusters');
        const data = await response.json();
        this.renderClusterSummary(data);
    }

    renderClusterSummary(data) {
    const container = document.getElementById('cluster-summary');
    
    container.innerHTML = `
        <h3>Cluster Distribution (${data.total_clusters} clusters, ${data.total_songs} songs)</h3>
        <div class="cluster-grid">
            ${data.clusters.map(cluster => `
                <div class="cluster-card" data-cluster-id="${cluster.cluster_id}" onclick="dashboard.selectCluster(${cluster.cluster_id})">
                    <div class="cluster-header">
                        <h4>Cluster ${cluster.cluster_id}</h4>
                        <span class="cluster-count">${cluster.song_count} songs (${cluster.percentage.toFixed(1)}%)</span>
                    </div>
                    <div class="distribution-bar">
                        <div class="distribution-fill" style="width: ${cluster.percentage}%;"></div>
                    </div>
                    <div class="cluster-features">
                        <div class="feature-mini">Tempo: ${cluster.mean_features.tempo.toFixed(1)}</div>
                        <div class="feature-mini">Energy: ${cluster.mean_features.energy.toFixed(3)}</div>
                        <div class="feature-mini">Loudness: ${cluster.mean_features.loudness.toFixed(3)}</div>
                        <div class="feature-mini">Valence: ${cluster.mean_features.valence.toFixed(3)}</div>
                        <div class="feature-mini">Danceability: ${cluster.mean_features.danceability.toFixed(3)}</div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}


    async selectCluster(clusterId) {
        this.selectedCluster = clusterId;
        
        // Highlight selected cluster
        document.querySelectorAll('.cluster-card').forEach(card => {
            card.classList.remove('selected');
        });
        document.querySelector(`[data-cluster-id="${clusterId}"]`).classList.add('selected');
        
        // Load cluster details
        const response = await fetch(`/api/cluster/${clusterId}`);
        const data = await response.json();
        this.renderClusterDetails(data);
        
        // Highlight in PCA visualization
        this.highlightClusterInVisualization(clusterId);
    }

    renderClusterDetails(data) {
    const container = document.getElementById('cluster-details');
    
    // Get all 5 features
    const meanFeatures = data.mean_features || data.summary.mean_features;
    
    container.innerHTML = `
        <div class="cluster-detail-header">
            <h3>Cluster ${data.cluster_id} Details</h3>
            <p>${data.songs.length} songs with similar audio characteristics</p>
        </div>
        
        <div class="cluster-features-profile">
            <h4>Average Feature Profile</h4>
            <div class="feature-bar-chart" id="feature-bar-chart"></div>
        </div>
        
        <div class="most-similar-songs">
            <h4>Most Representative Songs (Top 10)</h4>
            <div class="song-list">
                ${data.most_similar_songs.map((song, idx) => `
                    <div class="song-item">
                        <span class="song-rank">#${idx + 1}</span>
                        <span class="song-filename">${song.filename}</span>
                        <span class="song-genre">${song.genre}</span>
                        <span class="song-distance">Distance: ${song.distance_from_centroid.toFixed(4)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="all-songs-section">
            <button class="toggle-all-songs" onclick="dashboard.toggleAllSongs()">
                Show All ${data.songs.length} Songs
            </button>
            <div id="all-songs-list" class="song-list" style="display: none;">
                ${data.songs.map(song => `
                    <div class="song-item">
                        <span class="song-filename">${song.filename}</span>
                        <span class="song-genre">${song.genre}</span>
                        <span class="song-distance">Distance: ${song.distance_from_centroid.toFixed(4)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Render horizontal bar chart
    this.renderFeatureBarChart(meanFeatures);
}

renderFeatureBarChart(features) {
    const container = document.getElementById('feature-bar-chart');
    
    // All 5 features with proper labels
    const featureData = [
        { name: 'Tempo', key: 'tempo', maxValue: 200 },
        { name: 'Energy', key: 'energy', maxValue: 1 },
        { name: 'Loudness', key: 'loudness', maxValue: 1 },
        { name: 'Valence', key: 'valence', maxValue: 1 },
        { name: 'Danceability', key: 'danceability', maxValue: 1 }
    ];
    
    container.innerHTML = featureData.map(feature => {
        const value = features[feature.key] || 0;
        const percentage = (value / feature.maxValue) * 100;
        const displayValue = feature.maxValue === 1 ? value.toFixed(3) : value.toFixed(1);
        
        return `
            <div class="feature-bar-item">
                <div class="feature-bar-header">
                    <span class="feature-bar-label">${feature.name}</span>
                    <span class="feature-bar-value">${displayValue}</span>
                </div>
                <div class="feature-bar-container">
                    <div class="feature-bar-fill" style="width: ${percentage}%">
                        <span class="feature-bar-percentage">${percentage.toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}



    renderFeatureRadarChart(labels, values) {
    const canvas = document.getElementById('feature-radar-chart');
    
    // Destroy existing chart if it exists
    if (window.radarChartInstance) {
        window.radarChartInstance.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    // Ensure all 5 features are included
    const allFeatures = ['tempo', 'energy', 'loudness', 'valence', 'danceability'];
    const featureLabels = allFeatures.map(f => 
        f.charAt(0).toUpperCase() + f.slice(1)
    );
    
    // Map the values to ensure all 5 features have data
    const featureValues = allFeatures.map(feature => {
        const index = labels.indexOf(feature);
        return index !== -1 ? values[index] : 0;
    });
    
    // Normalize values for better visualization
    // Find the max value to scale appropriately
    const maxValue = Math.max(...featureValues);
    const scaleFactor = maxValue < 10 ? 100 : 1; // Scale if values are too small
    
    window.radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: featureLabels,
            datasets: [{
                label: 'Feature Values',
                data: featureValues.map(v => v * scaleFactor),
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 3,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)',
                pointRadius: 5,
                pointHoverRadius: 7,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.5,
            scales: {
                r: {
                    angleLines: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)',
                        lineWidth: 1
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)',
                        circular: true
                    },
                    pointLabels: {
                        font: {
                            size: 14,
                            weight: '600',
                            family: "'Inter', sans-serif"
                        },
                        color: '#1e293b',
                        padding: 15
                    },
                    ticks: {
                        display: true,
                        backdropColor: 'transparent',
                        color: '#64748b',
                        font: {
                            size: 11
                        },
                        stepSize: maxValue < 10 ? 20 : Math.ceil(maxValue / 5)
                    },
                    suggestedMin: 0,
                    suggestedMax: maxValue < 10 ? 100 : Math.ceil(maxValue * 1.2)
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const actualValue = context.parsed.r / scaleFactor;
                            return `${context.label}: ${actualValue.toFixed(3)}`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}
