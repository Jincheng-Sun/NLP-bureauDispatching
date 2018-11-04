import sklearn

import tensorflow as tf
import numpy as np
import pandas as pd
import csv

import math

Flags = tf.flags.FLAGS
# 定义网络参数
learning_rate = 0.001
display_step = 5
num_epochs = 20
keep_prob = 0.5
n_cls = 6
iters = 2000


def conv_layer(input, name, kh, kw, shape_in, shape_out, padding="SAME",usebias=True):
    # 卷积层
    input = tf.convert_to_tensor(input)
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, shape_in, shape_out],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding=padding)
        if(usebias):
            bias_init_val = tf.constant(0.0, shape=[shape_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            activation = tf.nn.leaky_relu(tf.nn.bias_add(conv, biases), alpha=0.1, name=scope)
        else:
            activation = tf.nn.leaky_relu(conv, alpha=0.1, name=scope)
    return activation


def fc_layer(input, name, shape_output):
    # 全连接层
    shape_input = input.get_shape()[-1].value
    in_reshape = tf.reshape(input, [-1, shape_input])
    with tf.name_scope(name) as scope:
        # kernel = tf.get_variable(scope + 'w',
        #                          shape=[shape_input, shape_output],#
        #                          dtype=tf.float32,
        #                          initializer=tf.truncated_normal_initializer(stddev=0.1)
        #                          )
        # print(kernel)
        # biases = tf.Variable(tf.constant(0.1, shape=[shape_output], dtype=tf.float32), name='b')
        # print(biases)
        # logits = tf.nn.relu_layer(in_reshape, kernel, biases)
        # logits = tf.add(tf.matmul(in_reshape,kernel),biases)
        # print(logits)
        logits = tf.layers.dense(in_reshape,shape_output,name=scope,activation=tf.nn.leaky_relu)
        print(logits)
    return logits


def nlp_structure(input):  # input: 152*152*1

    pool1 = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 76*76*1
    conv1 = conv_layer(pool1, "conv1", 3, 3, 1, 128,usebias=False)  # 76*76*128,W[1]=128*9
    pool2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # 38*38*128

    conv2 = conv_layer(pool2, "conv2", 3, 3, 128, 128, padding='VALID')  # 36*36*128 W[2]=128*128*9

    pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 18*18*128

    conv3 = conv_layer(pool3, "conv3", 3, 3, 128, 256, padding='VALID')  # 16*16*256 W[3]=128*256*9

    pool4 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 8*8*256

    conv4 = conv_layer(pool4, "conv4", 3, 3, 256, 512, padding='VALID')  # 6*6*512 W[4]=256*512*9
    conv5 = conv_layer(conv4, "conv5", 3, 3, 512, 256, padding='VALID')  # 4*4*256 W[5]=512*256*9
    conv6 = conv_layer(conv5, "conv6", 3, 3, 256, 128, padding='VALID')  # 2*2*128 W[6]=256*128*9

    pool5 = tf.nn.avg_pool(conv6, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    logits = fc_layer(pool5, "fc", 6)
    return logits


def read_dataset(file):
    file_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
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


def dataset_input_fn():
    filenames = './train.tfrecords'
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'vec_raw': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        vector = tf.decode_raw(parsed['vec_raw'], tf.float64)
        vector = tf.reshape(vector, [152, 152, 1])
        vector = tf.cast(vector, tf.float32)
        label = tf.cast(parsed['label'], tf.int32)
        label = tf.one_hot(label, n_cls, 1, 0)
        return vector, label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(50)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def train():
    batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 152, 152, 1], name='input')
    batch_y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')
    vec_batch, label_batch = dataset_input_fn()

    logits = nlp_structure(batch_x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(iters):
            v_batch, l_batch = sess.run([vec_batch, label_batch])

            _, loss_var, ac_var,logit= sess.run([optimizer, loss,accuracy,logits], feed_dict={batch_x: v_batch, batch_y: l_batch})
            # print(logit)
            # print(l_batch)

            print("Step [%d]  Loss : %f, training accuracy :  %g" % (i, loss_var, ac_var))
            if (i % 200 == 1):
                saver.save(sess, './model/model.ckpt', global_step=i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
