# Freya Kailing and Isabella Moppel
# CSCI 373
# Last updated 12/02/24

import pandas
import sklearn.cluster

SEED = 12345

# reads in data and labels and shuffles
def read_data(seed=SEED):
    data = pandas.read_csv("TCGA-PANCAN-HiSeq-801x20531/data.csv", index_col = 0)
    labels = pandas.read_csv("TCGA-PANCAN-HiSeq-801x20531/labels.csv", index_col = 0)

    # combines into one dataframe and shuffles
    data_with_labels = data.join(labels)
    shuffled = data_with_labels.sample(frac=1, random_state = seed)

    # splits labels ("Class") from rest of data
    data_shuffled = shuffled.drop("Class", axis=1)
    labels_shuffled = shuffled["Class"]

    return data_shuffled, labels_shuffled

data, labels = read_data()