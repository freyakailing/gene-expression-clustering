# Freya Kailing and Isabella Moppel
# CSCI 373
# Last updated 12/02/24

import pandas
import sklearn.cluster
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    data_2d = TSNE().fit_transform(data)

    # combines reduced data and labels in dataframe
    data_2d_labeled = labels.to_frame()
    data_2d_labeled["component_1"] = data_2d[:,0].tolist()
    data_2d_labeled["component_2"] = data_2d[:,1].tolist()

    print(data_2d_labeled.head())

    # graphs data in t-SNE embedded space
    plot = (
        ggplot(data_2d_labeled)
        + aes(x="component_1", y="component_2",shape="factor(label)", color="factor(label)")
        + geom_point()
        )
    plot.show()


# runs K-Means clustering with given number of clusters and returns Series with 
# the index of the cluster each sample belongs to
def kmeans(data, num_clusters, seed=SEED):
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data)
    labels = kmeans.labels_

    # converts labels to Series named "labels" to make display easier
    labels = pandas.Series(labels).rename("label")
    return labels


data, labels = read_data()
kmeans_labels = kmeans(data, 5)
graph_2d(data, kmeans_labels)