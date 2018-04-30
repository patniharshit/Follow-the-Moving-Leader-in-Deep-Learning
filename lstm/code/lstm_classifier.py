from dataModelLib import *
from dataCreate import get_train_test_data
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import optimizers
from keras_contrib.optimizers import ftml
import re
import random
import pandas as pd
import sys

path = "../data/"

_epochs = int(sys.argv[1])
def process_data(x_train, x_test):

    print "Size of x_train " + str(len(x_train)) 
    print "Size of x_test " + str(len(x_test)) 
    print "Processing x_train"
    for i, text in enumerate(x_train):
        text = preprocess(text)
        x_train[i] = stem(text, 0)
    print "Processing x_test"
    for i, text in enumerate(x_test):
        text = preprocess(text)
        x_test[i] = stem(text, 0)
    print "Processing completed.."
    return x_train, x_test


train_data_points = 2000 # each gender
max_features = 4000

x_train, y_train, x_test, y_test = get_train_test_data\
        (train_data_points)
x_train, x_test = process_data(x_train, x_test)

data = pd.DataFrame({'text':x_train, 'gender':y_train})
data_test = pd.DataFrame({'text':x_test, 'gender':y_test})
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

X_1 = tokenizer.texts_to_sequences(data_test['text'].values)
X_1 = pad_sequences(X_1)
s = len(X[0])
'''
np.temp = np.zeros((600, s))
temp = X_1[:, len(X_1[0]) - s:]
X_1 = temp
'''
Y = pd.get_dummies(data['gender']).values
Y_1 = pd.get_dummies(data_test['gender']).values
X_train, X_test, Y_train, Y_test = train_test_split(X_1,Y_1,test_size=0.33, random_state=42)
embed_dim = 196
lstm_out = 128
batch_size = 32

import matplotlib
import matplotlib.pyplot as plt

#ftml
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X_1.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer=ftml())
print(model.summary())
'''
X_train, Y_train = X, Y
X_test, Y_test = X_1, Y_1
'''
history = model.fit(X_train, Y_train, epochs=_epochs, batch_size=batch_size, verbose = 2)
plt.plot(history.history['loss'], 'r', label='FTML')

#adam
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X_1.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=_epochs, batch_size=batch_size, verbose = 2)
plt.plot(history.history['loss'],'b', label = 'ADAM')


#rmsprop
opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X_1.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer=opt)
history = model.fit(X_train, Y_train, epochs=_epochs, batch_size=batch_size, verbose = 2)
plt.plot(history.history['loss'], 'g', label = 'RMSProp')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()
