from ModelTraining.network import conv_layer,fc_layer,dataset_input_fn
from ModelTraining.network import nlp_structure
import tensorflow as tf
learning_rate = 0.001
display_step = 5
num_epochs = 20
keep_prob = 0.5
n_cls = 6
iters = 2000

def net_structure():
    pass

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
    writer=tf.summary.FileWriter('graph/tensorbord',sess.graph)
writer.close()