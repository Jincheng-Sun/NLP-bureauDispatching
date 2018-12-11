from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from tfidf import get_instance_tfidf_vector

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
def create_dataset():
    train_data = pd.read_csv('../datas/8w_trainset.csv', encoding='GB18030')

    train_data = train_data[:40000]
    train = pd.DataFrame({
        'ID': train_data['ID'],
        'Label': train_data[train_data.columns[-1]],
        'Feature': train_data['ID'].apply(lambda x: get_instance_tfidf_vector(str(x)))
    })
    x_train = np.array(train['Feature'].values.tolist())
    y_train = train['Label']
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)


num=40000
X_train = np.load('x_train.npy')[0:num]
Y_train = np.load('y_train.npy')[0:num]

x_train, X_test, y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
y_train = pd.DataFrame(y_train)[0]
y_val = pd.DataFrame(y_val)[0]
# one-hot，5 category
y_labels = list(y_train.value_counts().index)
#y_labels = np.unique(y_train)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), num_labels)

# model = Sequential()
# model.add(Embedding(x_train.shape[0], 3000, input_shape=(x_train.shape[1],)))
# model.add(Convolution1D(128, 5, activation='relu'))
# model.add(MaxPool1D(5, 5,padding='same'))
# model.add(Convolution1D(128, 5, activation='relu'))
# model.add(MaxPool1D(5, 5,padding='same'))
# model.add(Convolution1D(128, 5, activation='relu'))
# model.add(MaxPool1D(35, 35,padding='same'))
# model.add(Flatten())
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(5, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=1000,
#           epochs=10,
#           validation_data=(x_val, y_val))

model = Sequential()
model.add(Dense(1024, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=500,
          epochs=10,
          validation_data=(x_val, y_val))