import os
import tensorflow as tf
import pandas as pd
import numpy as np
from ModelTraining.tfidf import get_instance_tfidf_vector
import time
import csv
from itertools import islice
def cvt_label(label):
    array=np.array([[0,0,0,0,0,0]])
    if(label=='None'):
        array[0][5]=1
    else:
        array[0][int(label)]=1

    return array

def create_dataset(path):
    i = 0
    csvfile = csv.reader(open('../datas/testset_with_label.csv'))

    writer = tf.python_io.TFRecordWriter("train2000.tfrecords")
    writer2 = tf.python_io.TFRecordWriter("test200.tfrecords")

    for line in islice(csvfile,2,None):

        vec_id=line[0]
        # vector=vector.reshape(152,152)

        #根据id寻找pickle中对应的值
        vec=get_instance_tfidf_vector(vec_id)
        vec=np.array(vec)
        makeup=np.zeros(236)
        vec=np.concatenate([vec,makeup])
        vec=vec.reshape(152,152)
        vec_raw=vec.tobytes()
        label=line[10]
        if(label=='None' or label!=1):
            label=0
        label=np.int64(label)
        example=tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'vec_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vec_raw]))
        }))
        i += 1
        if(i<=2000):
            writer.write(example.SerializeToString())

        if(i%1000==0):
            print(i)
        if(i>2000):

            writer2.write(example.SerializeToString())

        if(i>2200):
            break
    writer.close()

def read_dataset(file,epochs):
    file_queue=tf.train.string_input_producer([file],num_epochs=epochs)
    reader=tf.TFRecordReader()

    _, serialized_example=reader.read(file_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={
                                         'label':tf.FixedLenFeature([],tf.int64),
                                         'vec_raw':tf.FixedLenFeature([],tf.string)
                                     })
    vector=tf.decode_raw(features['vec_raw'],tf.float64)

    vector=tf.reshape(vector,[152,152,1])
    vector=tf.cast(vector,tf.float32)
    label=tf.cast(features['label'],tf.int64)
    return vector,label


if __name__=='__main__':
    create_dataset('')



