import numpy as np
import pickle
import os
import pandas as pd
# import tensorflow as tf
# from itertools import islice
from synonyms import synonyms
from ModelTraining.tfidf import get_single_tfidf,get_instance_tfidf_vector

def scale(words,id):
    w2vs = np.array([])
    count = 0
    for word in words:
        try:
            # tfidf=get_single_tfidf(word,id)
            vector = synonyms.v(word)

        except KeyError:
            vector = np.zeros([1, 100])

        w2vs = np.append(w2vs, vector)
        count += 1
    w2vs = np.reshape(w2vs, [count, 100])
    w2vs = np.sum(w2vs, axis=0) / count
    return w2vs


def createDataset():
    instance_tokens_file = "tfidf_instance_tokens.pickle"

    with open(instance_tokens_file, 'rb') as itf:
        instance_tokens = pickle.load(itf)
    datacsv = pd.read_csv('../datas/4w_trainset.csv', encoding='gb18030')
    dataset = []

    # print(datacsv.head())

    for i in range(0, datacsv.shape[0]):
        id = datacsv.loc[i]['ID']
        words = instance_tokens[str(id)]
        words = scale(words,id)
        words = np.array(words)
        # words = words.flatten()
        label = datacsv.loc[i]['单位类别']
        dataset.append([id, words, label])
        # dataset[0][1].reshape([247, 100])
        if i % 1000 == 0:
            print(i)
        if ((i + 1) % 10000 == 0):
            df = pd.DataFrame(dataset, columns=['ID', 'Features', 'Labels'])
            ndata = np.array(df)
            np.save('dataset_4w%d.npy' % (i + 1), ndata)
            dataset.clear()


def loadDataset(i):
    features, labels = [], []
    npy1 = np.load('dataset_4w10000.npy')
    features = np.array(npy1[:, 1].tolist()).reshape([10000, 1, 100, 1])
    labels = np.array(npy1[:, 2].tolist())
    for n in range(2, i + 1):
        npy = np.load('dataset_4w%d0000.npy' % n)
        feature = npy[:, 1].tolist()

        feature = np.array(feature)

        feature = feature.reshape([10000, 1, 100, 1])

        label = npy[:, 2].tolist()

        label = np.array(label)

        features = np.append(features, feature, axis=0)

        labels = np.append(labels, label, axis=0)

    return features, labels


createDataset()
