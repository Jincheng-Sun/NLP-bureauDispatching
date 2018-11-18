from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras import regularizers
from keras import layers
from keras.utils import to_categorical
import tensorflow as tf
from keras_cnn.createDataset import loadDataset
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from keras import activations
#swich backend by changing ~ .keras/keras.json
from keras import backend as K

num_epochs=10
n_cls=5
learning_rate=0.001
batch_size=500

model=Sequential()
model.add(Conv2D(input_shape=[247,100,1],
                 name='conv1',
                 filters=16,
                 kernel_size=(3,3),
                 kernel_regularizer=regularizers.l1(),
                 padding='Same',
                 activation='elu'
                 )
          )

model.add(layers.MaxPool2D(pool_size=[2,2],strides=[2,2]))

model.add(Conv2D(name='conv2',
                 filters=32,
                 kernel_size=(3,3),
                 kernel_regularizer=regularizers.l1(),
                 padding='Same',
                 activation='elu'
                 )
          )

model.add(layers.AvgPool2D(pool_size=[8,4],strides=[8,4]))

model.add(layers.Flatten())

model.add(Dense(units=32,
                kernel_regularizer=regularizers.l2(),
                activity_regularizer=regularizers.l1(),
                activation='elu',
                )
          )

model.add(layers.BatchNormalization())

model.add(Dense(units=n_cls,
                kernel_regularizer=regularizers.l2(),
                activity_regularizer=regularizers.l1(),
                activation='softmax',
                )
          )
model.add(layers.BatchNormalization())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

dataX,dataY=loadDataset(4)
dataY=to_categorical(dataY)

# feature=
#
model.fit(x=dataX,y=dataY,batch_size=500,epochs=50)

