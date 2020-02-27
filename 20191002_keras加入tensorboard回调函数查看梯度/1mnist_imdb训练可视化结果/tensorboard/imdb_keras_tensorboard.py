# -*- coding: utf-8 -*-
"""
tensorboard_keras_imdb_Conv1D
"""
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing,layers,callbacks
from keras.models import Sequential
from keras.layers import Flatten, Dense

max_features = 2000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

tbCallBack = callbacks.TensorBoard(
                log_dir='imdb_logs',
                histogram_freq=1,
                #embeddings_freq=1
                )
history = model.fit(x_train, y_train,
                    epochs=3,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=2,
                    callbacks=[tbCallBack])