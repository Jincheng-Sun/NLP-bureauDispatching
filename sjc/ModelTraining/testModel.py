import tensorflow as tf
# from NLPproject import nlp_structure
import sys

sys.path.append("/Users/sunjincheng/Desktop/NLPproject/NLP-bureauDispatching/sjc/ModelTraining/")
from network import nlp_structure

def load_dataset():
    filenames = '../train36000.tfrecords'
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
        label = tf.one_hot(label, 6, 1, 0)
        return vector, label

    dataset = dataset.map(parser)
    dataset = dataset.batch(4000)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def test():
    batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 152, 152, 1], name='testinput')
    batch_y = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='testlabel')
    init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    output=nlp_structure(batch_x)
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    vec_batch, label_batch = load_dataset()
    with tf.Session() as sess:

        # saver = tf.train.import_meta_graph('model/model.ckpt-1801.meta')
        saver = tf.train.Saver()
        saver.restore(sess,'./model/model.ckpt-1801')
        sess.run(init)
        # for i in range(4000):
        v_batch,l_batch=sess.run([vec_batch,label_batch])
        print(v_batch.shape)
        pred=sess.run([accuracy],feed_dict={batch_x:v_batch,batch_y:l_batch})

        print(pred)


if __name__ == '__main__':
    test()





