from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans_clustering(X_numeric, n_clusters=3):
    """Scales numeric data and runs KMeans clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, scaler
