# Freya Kailing and Isabella Moppel
# CSCI 373
# Last updated 12/05/24

import pandas
import sklearn.cluster
import sklearn.mixture
from plotnine import *
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist,squareform

SEED = 12345

# K Means
def perform_KMeans (data):
    silhouettes_KMeans = kmeans_silhouettes(80, data)
    best_cluster_KMeans = findBestClusterValue(silhouettes_KMeans)
    kmeans_labels = kmeans(data, best_cluster_KMeans)
    graph_2d(data, kmeans_labels, "K Means Clusters")

# Spectral clustering
def perform_spectral(data):
    silhouettes = spectral_silhouettes(20, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    spectral_clusters = spectral_clustering(data, best_cluster_num)
    graph_2d(data, spectral_clusters, "Spectral Clusters")

# Gaussain mixture labelling
def perform_gaussian(data):
    silhouettes = gaussian_silhouettes(20, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    gaussian_clusters = gaussian_mixture(data, best_cluster_num)
    graph_2d(data, gaussian_clusters, "Gaussian Mixture Clusters")

# reads in data and labels and shuffles
def read_data(seed=SEED):
    data = pandas.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col = 0)
    labels = pandas.read_csv("TCGA-PANCAN-HiSeq-801x20531/labels.csv", index_col = 0)

    # combines into one dataframe and shuffles
    data_with_labels = data.join(labels)
    shuffled = data_with_labels.sample(frac=1, random_state = seed)

    # splits labels from rest of data
    data_shuffled = shuffled.drop("Class", axis=1)
    labels_shuffled = shuffled["Class"]
    labels_shuffled = labels_shuffled.rename("label") #uses "label" name for consistency

    return data_shuffled, labels_shuffled

# visualizes data in two dimensions, colored by actual class or cluster
# (requires DataFrame for "data", Series or DataFrame for "labels")
def graph_2d(data, labels, title):
    # TODO maybe: use principal component analysis to reduce to ~50 attributes
    # (recommended in t-SNE documentation but it looks like it's doing good as-is)

    # uses t-SNE to reduce to 2 components
    data_2d = TSNE().fit_transform(data)

    # combines reduced data and labels in dataframe
    data_2d_labeled = labels.to_frame()
    data_2d_labeled["component_1"] = data_2d[:,0].tolist()
    data_2d_labeled["component_2"] = data_2d[:,1].tolist()

    # graphs data in t-SNE embedded space
    plot = (
        ggplot(data_2d_labeled)
        + aes(x="component_1", y="component_2",shape="factor(label)", color="factor(label)")
        + geom_point()
        )
    plot.show()
    plot.save(title)

# runs K-Means clustering with given number of clusters and returns Series with 
# the index of the cluster each sample belongs to
def kmeans(data, num_clusters, seed=SEED):
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data)
    labels = kmeans.labels_

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels

# runs K-Means clustering with a range of values for number of clusters and prints and returns silhouette scores
def kmeans_silhouettes (maxClusterValue, X, seed=SEED):

    range_n_clusters = []

    for i in range (2, maxClusterValue+1):
        range_n_clusters.append(i)

    silhouettes = {}
    for n_clusters in range_n_clusters:

        clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        silhouettes[n_clusters] = silhouette_avg
    
    return silhouettes

# runs spectral clustering for a given number of clusters, returns Series of cluster indices
def spectral_clustering(data, num_clusters):
    # constructs affinity matrix by computing a graph of nearest neighbors, then applies clustering methods
    spectral = sklearn.cluster.SpectralClustering(affinity='nearest_neighbors', n_clusters=num_clusters, random_state=SEED)
    labels = spectral.fit_predict(data)

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels

# runs spectral clustering with a range of values for number of clusters and prints and returns silhouette scores
def spectral_silhouettes (maxClusterValue, X, seed=SEED):
    range_n_clusters = list(range(2, maxClusterValue+1))

    silhouettes = {}
    for n_clusters in range_n_clusters:
        cluster_labels = spectral_clustering(X, n_clusters)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"n_clusters: {n_clusters}, Average silhouette_score: {silhouette_avg}")
        silhouettes[n_clusters] = silhouette_avg
    
    return silhouettes

# runs a gaussian mixture estimation for a given number of clusters, returns Series of cluster indices
def gaussian_mixture(data, num_clusters):
    # diagonal covariance used for speed (compared to full)
    gaussian = sklearn.mixture.GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=SEED)
    labels = gaussian.fit_predict(data)

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels

# runs gaussian mixture clustering with a range of values for num_clusters and prints and returns silhouette scores
def gaussian_silhouettes (maxClusterValue, X, seed=SEED):
    range_n_clusters = list(range(2, maxClusterValue+1))

    silhouettes = {}
    for n_clusters in range_n_clusters:
        cluster_labels = gaussian_mixture(X, n_clusters)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"n_clusters: {n_clusters}, Average silhouette_score: {silhouette_avg}")
        silhouettes[n_clusters] = silhouette_avg
    
    return silhouettes

def findBestClusterValue (silhouettes):
    max_cluster = 2
    for cluster in silhouettes:
        if silhouettes[cluster] > silhouettes[max_cluster]:
            max_cluster = cluster
    return max_cluster

data, labels = read_data()