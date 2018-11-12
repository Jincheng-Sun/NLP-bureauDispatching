from ModelTraining.network import dataset_input_fn
from ModelTraining.network import nlp_structure
import tensorflow as tf

learning_rate = 0.005
display_step = 5
num_epochs = 1
keep_prob = 0.5
n_cls = 6
iters = 3600

def conv_layer(input, name, kh, kw, shape_in, shape_out, padding="SAME",usebias=True,istraining=True):
    # 卷积层
    input = tf.convert_to_tensor(input)
    with tf.name_scope(name) as scope:
        #初始化kernel，kh，kw为卷积核尺寸，shape_in为卷积前通道数，shape_out为卷积后通道数，shape_in*shape_out即卷积核数量
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, shape_in, shape_out],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.2),
                                 trainable=True
                                 )
        #卷积操作，步长为[1,1]，即kernel*input
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding=padding)
        #对卷积后的数据进行BatchNormalization操作
        bn = tf.layers.batch_normalization(conv,training=istraining,momentum=0.9)

        #添加bias，对结果进行activation
        if(usebias):
            bias_init_val = tf.constant(0.0, shape=[shape_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            activation = tf.nn.leaky_relu(tf.nn.bias_add(bn, biases), alpha=0.1, name=scope)
            #tensorboard直方图记录权值，bias变化
            tf.summary.histogram('w', kernel)
            tf.summary.histogram('b', biases)
        else:
            activation = tf.nn.leaky_relu(bn, alpha=0.1, name=scope)
            tf.summary.histogram('w', kernel)
    return activation


def fc_layer(input, name, shape_output,usebias=True,istraining=True,useactivation=True,dropout=0.0):
    # 全连接层
    #shape_input为每个feature flatten后的size
    shape_input = input.get_shape()[-1].value
    #reshape成[batch_size,shape_input]的size
    in_reshape = tf.reshape(input, [-1, shape_input])
    with tf.name_scope(name) as scope:
        #初始化kernel
        kernel = tf.get_variable(scope + 'w',
                                 shape=[shape_input, shape_output],#
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.2),
                                 trainable=True
                                 )

        mul = tf.matmul(in_reshape, kernel)
        bn = tf.layers.batch_normalization(mul, training=istraining,momentum=0.9)
        if(usebias):
            biases = tf.Variable(tf.constant(0.1, shape=[shape_output], dtype=tf.float32), name='b')
            logits = tf.add(bn, biases)
            tf.summary.histogram('w', kernel)
            tf.summary.histogram('b', biases)
        else:
            logits=bn
            tf.summary.histogram('w', kernel)

        if(useactivation):
            logits = tf.nn.leaky_relu(logits)
        if(dropout!=0):
            logits = tf.nn.dropout(logits,keep_prob=dropout)

        # w=tf.get_default_graph().get_tensor_by_name()
    return logits
def model(input,usebias=False,istraining=True):
    #Layer 0：input size:[m,152,152,1],经过第一层pool,output size [m,38,38,1]
    pool1 = tf.nn.avg_pool(input, [1, 4, 4, 1], [1, 4, 4, 1], padding='VALID', name='pool1')
    #Layer 1：经过第一层conv, kernel [3*3,1*32], output size [m,38,38,32],
    conv1 = conv_layer(pool1, 'conv1', 3, 3, 1, 32,usebias=usebias,istraining=istraining)
    #Layer 2：第二层pooling，output size [m,19,19,32]
    pool2 = tf.nn.avg_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    #Layer 3：第二层卷积, kernel [3*3,32*32], output size [m,19,19,32]
    conv2 = conv_layer(pool2, 'conv2', 3, 3, 32, 32,usebias=usebias,istraining=istraining)
    #Flatten, output size [m,1,1,19*19*32]
    conv2 = tf.reshape(conv2,[-1,1,1,11552])
    #Layer 4：第一层全连接层, kernel [11552,256],output size [m, 256]
    fc1 = fc_layer(conv2, 'fc1', 256,usebias=usebias,istraining=istraining,dropout=0.5)
    #Layer 5：第二层全连接层, kernel [256,64], output size [m,64]
    fc2 = fc_layer(fc1, "fc2", 64,usebias=usebias,istraining=istraining,dropout=0.5)
    #Layer 6：第三层全连接层, kernel [64,n_cls], output size [m,n_cls], 不使用激活函数
    fc3 = fc_layer(fc2, 'fc3', n_cls,usebias=usebias,useactivation=False,istraining=istraining,dropout=0.5)
    # logits = tf.nn.softmax(fc3)
    logits = fc3
    return logits

def onlyFcModel(input):
    input=tf.layers.flatten(input)
    # fc1 = fc_layer(input,'fc1',256,usebias=False)
    # l1=tf.layers.dropout(fc1)
    fc2 = fc_layer(input,'fc3',n_cls)
    l2=tf.layers.dropout(fc2)
    logits=tf.nn.softmax(l2)
    return logits

def train():
    with tf.name_scope('input'):
        batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 152, 152, 1], name='input')
        batch_y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')
    vec_batch, label_batch = dataset_input_fn('../train36000.tfrecords', 10, 400)

    logits=model(batch_x,usebias=True)


    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(extra_update_ops):
        update_op = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))


    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()

    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('normalization_loss',n_loss)
    tf.summary.scalar('accuracy', accuracy)
    merge = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter('logs/', sess.graph)

        for i in range(iters):
            v, l = sess.run([vec_batch, label_batch])

            _, LOSS, ACCURACY, MERGE,log = sess.run([update_op, loss, accuracy, merge,logits], feed_dict={batch_x: v, batch_y: l})
            # print('真实值：'+str(l))
            # print('估计值：'+str(log))

            if (i > 0 and i % 10 == 0):
                writer.add_summary(MERGE, i)
                print("Step [%d]  Loss : %f, training accuracy :  %g" % (i, LOSS, ACCURACY))

            if (i > 0 and i % 100 == 0):
                saver.save(sess, './model/model.ckpt', global_step=i)
        coord.request_stop()
        # coord.join(threads)

def test():
    batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 152, 152, 1], name='testinput')
    batch_y = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='testlabel')
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    vec_batch, label_batch = dataset_input_fn('../test4000.tfrecords', 4000, 4000)

    logits=model(batch_x,istraining=False)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:

        # saver = tf.train.import_meta_graph('model/model.ckpt-1801.meta')
        saver = tf.train.Saver()
        saver.restore(sess,'./model/model.ckpt-3500')
        sess.run(init)
        # for i in range(4000):
        v_batch,l_batch=sess.run([vec_batch,label_batch])
        pred=sess.run([accuracy],feed_dict={batch_x:v_batch,batch_y:l_batch})

        print(pred)



if __name__=='__main__':
    test()