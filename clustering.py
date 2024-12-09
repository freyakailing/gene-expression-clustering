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
from sklearn.cluster import DBSCAN
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import rand_score

SEED = 12345
BEST_EPS = 0

# K Means
def perform_KMeans (data, labels):
    silhouettes = kmeans_silhouettes(80, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    kmeans_clusters = kmeans(data, best_cluster_num)
    graph_2d(data, kmeans_clusters, "K Means Clusters")
    mi_score = mutual_info_score(labels, kmeans_clusters)
    rscore = rand_score(labels, kmeans_clusters)
    print("KMeans has a mutual information score of " + mi_score + " and a rand score of " + rscore )

# Spectral clustering
def perform_spectral(data, labels):
    silhouettes = spectral_silhouettes(20, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    spectral_clusters = spectral_clustering(data, best_cluster_num)
    graph_2d(data, spectral_clusters, "Spectral Clusters")
    mi_score = mutual_info_score(labels, spectral_clusters)
    rscore = rand_score(labels, spectral_clusters)
    print("Spectral clustering has a mutual information score of " + mi_score + " and a rand score of " + rscore)

# Gaussian mixture labelling
def perform_gaussian(data, labels):
    silhouettes = gaussian_silhouettes(20, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    gaussian_clusters = gaussian_mixture(data, best_cluster_num)
    graph_2d(data, gaussian_clusters, "Gaussian Mixture Clusters")
    mi_score = mutual_info_score(labels, gaussian_clusters)
    rscore = rand_score(labels, gaussian_clusters)
    print("Gaussian mixture labelling has a mutual information score of " + mi_score + " and a rand score of " + rscore)

# DBSCAN
def perform_dbscan(data, labels):
    silhouettes = dbscan_silhouettes(20, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    dbscan_clusters = dbscan(data, best_cluster_num)
    graph_2d(data, dbscan_clusters, "DBSCAN Clusters")
    mi_score = mutual_info_score(labels, dbscan_clusters)
    rscore = rand_score(labels, dbscan_clusters)
    print("DBSCAN has a mutual information score of " + mi_score + " and a rand score of " + rscore)

# Mini Batch K Means
def perform_mini_batch_KMeans (data, labels):
    silhouettes = mini_batch_kmeans_silhouettes(80, data)
    best_cluster_num = findBestClusterValue(silhouettes)
    mini_kmeans_clusters = mini_batch_kmeans(data, best_cluster_num)
    graph_2d(data, mini_kmeans_clusters, "Mini Batch K Means Clusters")
    mi_score = mutual_info_score(labels, mini_kmeans_clusters)
    rscore = rand_score(labels, mini_kmeans_clusters)
    print("Mini batch KMeans has a mutual information score of " + mi_score + " and a rand score of " + rscore)


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


# runs dbscan estimation given a certain min_samples
def dbscan(data, min_samples):
    # run dbscan
    labels = sklearn.cluster.dbscan(eps = BEST_EPS, min_samples = min_samples).fit_predict(data)
    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels

# runs dbscan clustering with a range of values for num_clusters and a range of values for eps;
#  prints and returns silhouette scores
def dbscan_silhouettes (max_min_samples, X): # no seed
    range_min_samples = list(range(2, max_min_samples+1))

    # eps determines how far apart points need to be in order to be considered distinct. floats
    range_eps = list(range(1, 11))
    range_eps = [ep / 10.0 for ep in range_eps]  # eps will range from 0.1 to 1.0
    print(range_eps)

    silhouettes = {}

    for min_sample in range_min_samples:
        print( str(min_sample) + "\n")
        for ep in range_eps:
            cluster_labels = DBSCAN(eps = ep, min_samples = min_sample).fit_predict(X)
            print(cluster_labels)
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) # a label of -1 -> noise
            print(num_clusters)
            if num_clusters > 1: # silhouette score will only work if there are at least 2 clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                print(f"min sample: {min_sample}, eps: {ep}, Average silhouette_score: {silhouette_avg}")
                if silhouette_avg > silhouettes[min_sample]: # choose the best value of eps to put in
                    silhouettes[min_sample] = silhouette_avg
                    BEST_EPS = ep # save the best value of eps to use later

    return silhouettes


# runs Mini-K-Means clustering with given number of clusters and returns Series with 
# the index of the cluster each sample belongs to
def mini_batch_kmeans(data, num_clusters, seed=SEED):
    mini_kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, random_state=seed)
    mini_kmeans.fit(data)
    labels = mini_kmeans.labels_

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels


# runs Mini Batch K-Means clustering with a range of values for number of clusters and prints and returns silhouette scores
def mini_batch_kmeans_silhouettes (maxClusterValue, X, seed=SEED):

    range_n_clusters = []

    for i in range (2, maxClusterValue+1):
        range_n_clusters.append(i)

    silhouettes = {}
    for n_clusters in range_n_clusters:

        clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        silhouettes[n_clusters] = silhouette_avg
    
    return silhouettes


def findBestClusterValue (silhouettes):
    max_cluster = 2
    for cluster in silhouettes:
        if silhouettes[cluster] > silhouettes[max_cluster]:
            max_cluster = cluster
    return max_cluster

data, labels = read_data()

# perform_dbscan(data)
perform_KMeans(data, labels)
perform_spectral(data, labels)
perform_gaussian(data, labels)
perform_mini_batch_KMeans(data, labels)