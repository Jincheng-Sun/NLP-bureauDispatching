import os
import tensorflow as tf
import pandas as pd
import numpy as np
# from ModelTraining.tfidf import get_instance_tfidf_vector
import time
import csv
from itertools import islice
from synonyms import synonyms
import pickle


def cvt_label(label):
    array = np.array([[0, 0, 0, 0, 0, 0]])
    if (label == 'None'):
        array[0][5] = 1
    else:
        array[0][int(label)] = 1

    return array


# def create_dataset(path):
#     i = 0
#     csvfile = csv.reader(open('../datas/testset_with_label.csv'))
#
#     writer = tf.python_io.TFRecordWriter("r_train36000new.tfrecords")
#     writer2 = tf.python_io.TFRecordWriter("r_test4000new.tfrecords")
#
#     for line in islice(csvfile, 2, None):
#
#         vec_id = line[0]
#         # vector=vector.reshape(152,152)
#
#         # 根据id寻找pickle中对应的值
#         vec = get_instance_tfidf_vector(vec_id)
#         vec = np.array(vec)
#         makeup = np.zeros(236)
#         vec = np.concatenate([vec, makeup])
#         vec = vec.reshape(152, 152)
#         vec_raw = vec.tobytes()
#         label = line[10]
#         if (label == 'None'):
#             label = 0
#         label = np.int64(label)
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#             'vec_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vec_raw]))
#         }))
#         i += 1
#         if (i <= 36000):
#             writer.write(example.SerializeToString())
#
#         if (i % 1000 == 0):
#             print(i)
#         if (i > 36000 and i <= 40000):
#             writer2.write(example.SerializeToString())
#
#         if (i == 40000):
#             break
#     writer.close()


def read_dataset(file, epochs):
    file_queue = tf.train.string_input_producer([file], num_epochs=epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'vec_raw': tf.FixedLenFeature([], tf.string)
                                       })
    vector = tf.decode_raw(features['vec_raw'], tf.float64)

    vector = tf.reshape(vector, [152, 152, 1])
    vector = tf.cast(vector, tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return vector, label


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


def create_w2v_dataset():
    i = 0
    index_file = "tfidf_inverse_index.pickle"
    instance_tokens_file = "tfidf_instance_tokens.pickle"
    with open(index_file, 'rb') as iif:
        tfidf_inverse_index = pickle.load(iif)

    with open(instance_tokens_file, 'rb') as itf:
        instance_tokens = pickle.load(itf)

    csvfile = csv.reader(open('../datas/4w_trainset.csv',encoding='gb18030'))

    writer = tf.python_io.TFRecordWriter("train36000v1.tfrecords")
    writer2 = tf.python_io.TFRecordWriter("test4000v1.tfrecords")

    # for line in instance_tokens:
    #     id = line  # id
    #     words = instance_tokens[line]  # words:['美兰区千家村社区', '户口', '户口', '路', '沙', '美兰区', '中心', '人员', '业务', '人', '人才', '业务', '市', '公安局']
    #     vec = scale(words)
    for line in islice(csvfile, 1, None):

        vec_id = line[0]
        try:
            words= instance_tokens[vec_id]
        except KeyError:
            print('No.%s was not found' %KeyError)
            words = 0
            continue
        vector= scale(words)
        vec_raw=vector.tobytes()
        # vector=vector.reshape(152,152)

        # 根据id寻找pickle中对应的值
        # vec = get_instance_tfidf_vector(vec_id)
        # vec = np.array(vec)
        # makeup = np.zeros(236)
        # vec = np.concatenate([vec, makeup])
        # vec = vec.reshape(152, 152)
        # vec_raw = vec.tobytes()
        label = line[9]
        if (label == 'None'):
            print("None value")
        label = np.int64(label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'vec_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vec_raw]))
        }))
        i += 1
        if (i <= 72000):
            writer.write(example.SerializeToString())

        if (i % 1000 == 0):
            print(i)
        if (i > 72000 and i <= 80000):
            writer2.write(example.SerializeToString())

        if (i == 80000):
            break
    writer.close()

def read_w2v_dataset(file, epochs):
    file_queue = tf.train.string_input_producer([file], num_epochs=epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'vec_raw': tf.FixedLenFeature([], tf.string)
                                       })
    vector = tf.decode_raw(features['vec_raw'], tf.float64)

    # vector = tf.reshape(vector, [152, 152, 1])
    vector = tf.cast(vector, tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return vector, label

if __name__ == '__main__':
    create_w2v_dataset()