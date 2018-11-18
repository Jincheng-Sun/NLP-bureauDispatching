# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import pandas as pd
# import tensorflow as tf
#from itertools import islice
from synonyms import synonyms


def scale(words):
    w2vs = np.zeros([247, 100])
    count = 0
    for word in words:
        try:
            vector = synonyms.v(word)

        except KeyError:
            vector = np.zeros([1, 100])

        w2vs[count] = vector
        count += 1
    return w2vs

def createDataset():
    instance_tokens_file = "tfidf_instance_tokens.pickle"

    with open(instance_tokens_file, 'r') as itf:
        instance_tokens = pickle.load(itf)
    datacsv = pd.read_csv('../datas/4w_trainset.csv', encoding='gb18030')
    dataset = []

    # print(datacsv.head())

    for i in range(0, datacsv.shape[0]):
        id = datacsv.loc[i]['ID']
        words = instance_tokens[str(id)]
        words = scale(words)
        words = np.array(words)
        words = words.flatten()
        label = datacsv.loc[i]['单位类别']
        dataset.append([id, words, label])
        # dataset[0][1].reshape([247, 100])
        if i % 1000 == 0:
            print(i)
    df = pd.DataFrame(dataset, columns=['ID', 'Features', 'Labels'])
    ndata = np.array(df)
    # np.save('dataset_4w.npy', ndata)
    return ndata





def readDataset():
    csv=pd.read_csv('dataset_4w.csv',encoding='gb18030')
    print(csv.loc[0][1])



createDataset()