from keras import Sequential
#swich backend by changing ~ .keras/keras.json
from keras import backend as K
import tensorflow as tf

train_url = 'https://download.mlcc.google.cn/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)

print(train_path)

def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),  # terms are strings of varying lengths
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)  # labels are 0 or 1
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels


def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(3, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")#50

terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)
filename='train.tfrecord'
words,label=_input_fn(train_path)

feature_columns = [ terms_feature_column ]
classifier = tf.estimator.LinearClassifier(
  feature_columns=feature_columns,
)
with tf.Session() as sess:
    words_dict,label=sess.run([words,label])
    #words_dict[m,462]
    #在LinearClassifier中，words——dict中一个words->[462,50]