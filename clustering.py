# Freya Kailing and Isabella Moppel
# CSCI 373
# Last updated 12/05/24

import pandas
import sklearn.cluster
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

SEED = 12345

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
def graph_2d(data, labels):
    # TODO maybe: use principal component analysis to reduce to ~50 attributes
    # (recommended in t-SNE documentation but it looks like it's doing good as-is)

    # uses t-SNE to reduce to 2 components
    data_2d = TSNE(random_state=SEED).fit_transform(data)

    # combines reduced data and labels in dataframe
    data_2d_labeled = labels.to_frame()
    data_2d_labeled["component_1"] = data_2d[:,0].tolist()
    data_2d_labeled["component_2"] = data_2d[:,1].tolist()\

    # graphs data in t-SNE embedded space
    plot = (
        ggplot(data_2d_labeled)
        + aes(x="component_1", y="component_2",shape="factor(label)", color="factor(label)")
        + geom_point()
        )
    
    return plot


# runs K-Means clustering with given number of clusters and returns Series with 
# the index of the cluster each sample belongs to
def kmeans(data, num_clusters, seed=SEED):
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data)
    labels = kmeans.labels_

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels

def calculateSilhouetteScore (range_n_clusters, X, seed=SEED):
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

# finds the best number of clusters for k-Means algorithm
def test_kMeans():
    range_n_clusters = []

    for i in range (2, 81):
        range_n_clusters.append(i)

    silhouettes = calculateSilhouetteScore(range_n_clusters, data)

    max_cluster = 2;
    for cluster in silhouettes:
        if silhouettes[cluster] > silhouettes[max_cluster]:
            max_cluster = cluster
    
    return max_cluster

data, labels = read_data()

kmeans_labels = kmeans(data, 6)
plot = graph_2d(data, labels)
plot.save("Samples by Cancer Type")