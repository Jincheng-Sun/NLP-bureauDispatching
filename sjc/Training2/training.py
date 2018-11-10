from ModelTraining.network import conv_layer, fc_layer, dataset_input_fn
from ModelTraining.network import nlp_structure
import tensorflow as tf

learning_rate = 0.001
display_step = 5
num_epochs = 20
keep_prob = 0.5
n_cls = 6
iters = 2000

def model(input):
    # with tf.name_scope('pool1'):
    pool1 = tf.nn.avg_pool(input, [1, 4, 4, 1], [1, 4, 4, 1], padding='VALID', name='pool1')  # 152*152-->19*19
    # pool1 = tf.nn.avg_pool(batch_x, [1, 8, 8, 1], [1, 8, 8, 1], padding='VALID', name='pool1')

# with tf.name_scope('conv1'):
    conv1 = conv_layer(pool1, 'conv1', 3, 3, 1, 32)

# with tf.name_scope('pool2'):
    pool2 = tf.nn.avg_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

# with tf.name_scope('conv2'):
    conv2 = conv_layer(pool2, 'conv2', 3, 3, 32, 32)
    conv2 = tf.reshape(conv2,[-1,1,1,11552])
# with tf.name_scope('fc1'):
    fc1 = fc_layer(conv2, 'fc1', 256)

# with tf.name_scope('fc2'):
    fc2 = fc_layer(fc1, "fc2", 64)

# with tf.name_scope('fc3'):
    fc3 = fc_layer(fc2, 'fc3', n_cls)
    logits = tf.nn.softmax(fc3)
    return logits

def train():
    with tf.name_scope('input'):
        batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 152, 152, 1], name='input')
        batch_y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')
    vec_batch, label_batch = dataset_input_fn('../test4000.tfrecords', 36, 400)

    logits=model(batch_x)
    print(logits.shape)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merge = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter('logs/', sess.graph)

        for i in range(iters):
            v, l = sess.run([vec_batch, label_batch])

            _, LOSS, ACCURACY, MERGE = sess.run([optimizer, loss, accuracy, merge], feed_dict={batch_x: v, batch_y: l})
            if (i > 0 and i % 10 == 0):
                writer.add_summary(MERGE, i)
                print("Step [%d]  Loss : %f, training accuracy :  %g" % (i, LOSS, ACCURACY))

            if (i > 0 and i % 100 == 0):
                saver.save(sess, './model/model.ckpt', global_step=i)
        coord.request_stop()
        # coord.join(threads)

if __name__=='__main__':
    train()