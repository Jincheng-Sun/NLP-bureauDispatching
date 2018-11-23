
import tensorflow as tf
import math

learning_rate = 0.001
display_step = 5
num_epochs = 50
keep_prob = 0.5
n_cls = 5
iters = 3600
def conv_layer(input, name, kh, kw, shape_in, shape_out, padding="SAME",usebias=True,istraining=True):
    # 卷积层
    input = tf.convert_to_tensor(input)
    with tf.name_scope(name) as scope:
        #初始化kernel，kh，kw为卷积核尺寸，shape_in为卷积前通道数，shape_out为卷积后通道数，shape_in*shape_out即卷积核数量
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, shape_in, shape_out],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean=0.2,stddev=0.5),
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
            activation = tf.nn.leaky_relu(tf.nn.bias_add(conv, biases), alpha=0.1, name=scope)
            #tensorboard直方图记录权值，bias变化
            tf.summary.histogram('w', kernel)
            tf.summary.histogram('b', biases)
        else:
            activation = tf.nn.leaky_relu(conv, alpha=0.1, name=scope)
            tf.summary.histogram('w', kernel)
    return activation


def fc_layer(input, name, shape_output,usebias=True,istraining=True,useactivation=True,dropout=1):
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
                                 # initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.5),
                                 initializer=tf.initializers.lecun_normal(),
                                 trainable=True
                                 )

        mul = tf.matmul(in_reshape, kernel)
        bn = tf.layers.batch_normalization(mul, training=istraining,momentum=0.9)
        if(usebias):
            biases = tf.Variable(tf.constant(0, shape=[shape_output], dtype=tf.float32), name='b')
            logits = tf.add(mul, biases)
            tf.summary.histogram('w', kernel)
            tf.summary.histogram('b', biases)
        else:
            logits=mul
            tf.summary.histogram('w', kernel)

        if(useactivation):
            logits = tf.nn.leaky_relu(logits)
        if(dropout!=1):
            logits = tf.nn.dropout(logits,keep_prob=dropout)

        # w=tf.get_default_graph().get_tensor_by_name()
    return logits

def model(input):
    pass
def dataset(filenames,batch,buffersize,onehot=True):

    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'vec_raw': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        vector = tf.decode_raw(parsed['vec_raw'], tf.float64)
        # vector = sk.preprocessing.scale(vector)
        vector = tf.reshape(vector, [10,10,1])

        vector = tf.cast(vector, tf.float32)
        label = tf.cast(parsed['label'], tf.int32)
        if onehot:
            label = tf.one_hot(label, n_cls, 1, 0)
        else:
            label = tf.reshape(label,[1])
        return vector, label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=buffersize)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


batch_x = tf.placeholder(dtype=tf.float32, shape=[None, 10,10,1], name='testinput')
batch_y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='testlabel')


vector,label=dataset('train10000v1.tfrecords',1,1)
norm_input=tf.nn.l2_normalize(vector)
flat=tf.reshape(norm_input,shape=[-1,1,1,100])
fc2=fc_layer(flat,'fc2',n_cls,useactivation=False)
activation=tf.nn.softmax(fc2)

logits=activation

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=activation,labels=batch_y))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=batch_y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    update_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init= tf.group(tf.local_variables_initializer(),tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
tf.summary.scalar('loss', loss)
# tf.summary.scalar('normalization_loss',n_loss)
tf.summary.scalar('accuracy', accuracy)
merge = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()


    for i in range(iters):
        writer = tf.summary.FileWriter('logs/', sess.graph)
        vectors, labels = sess.run([vector, label])
        # n = sess.run(norm_input,feed_dict={batch_x:vectors})
        # f = sess.run(fc1)
        # act = sess.run(activation)
        _,LOSS,ACCURACY,MERGE,n,fl,fc,act=sess.run([update_op,loss,accuracy,merge,norm_input,flat,fc2,activation],feed_dict={batch_x:vectors,batch_y:labels})
        if(i==31):
            print(vectors)
            print(n)

            print(fl)
            print(fc)
            print(act)

        if (i > 0 and i % 10 == 0):
            writer.add_summary(MERGE, i)
            print("Step [%d]  Loss : %f, training accuracy :  %g" % (i, LOSS, ACCURACY))

        if (i > 0 and (i+1) % 100 == 0):
            saver.save(sess, './model/model.ckpt', global_step=i)
    coord.request_stop()