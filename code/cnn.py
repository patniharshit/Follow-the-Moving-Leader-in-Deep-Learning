import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Activation, ZeroPadding2D,Input,Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras_contrib.optimizers import ftml
import numpy as np
import pandas as pd

data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
X = np.array(data_train.iloc[:, 1:])
X = X.reshape(X.shape[0], img_rows, img_cols, 1)
y = to_categorical(np.array(data_train.iloc[:, 0:1]))

#X = X[30000:,:]
#y = y[30000:,:]

X_test = np.array(data_test.iloc[:, 1:])
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
y_test = to_categorical(np.array(data_test.iloc[:, 0:1]))

X = X.astype('float32')
X_test = X_test.astype('float32')
X /= 255
X_test /= 255

def train_model(shape=(28, 28, 1), num_classes=10):
    X_input = Input(shape=shape)
    X = Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               kernel_initializer='he_normal',
               input_shape=input_shape)(X_input)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(0.3)(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X, name="CNN")
    return model

epochs = 50

print('rmsprop start')
opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model = train_model()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X,y,batch_size=128, epochs=epochs, verbose=0)
plt.plot(history.history['loss'])
print('rmsprop end')

print('adadelta start')
model = train_model()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
history = model.fit(X,y,batch_size=128, epochs=epochs, verbose=0)
plt.plot(history.history['loss'])
print('adadelta end')

print('adam start')
model = train_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X,y,batch_size=128, epochs=epochs, verbose=0)
plt.plot(history.history['loss'])
print('adam end')

print('ftml start')
x_train, y_train = X, y
model = train_model()
model.compile(loss='categorical_crossentropy',
              optimizer=ftml(beta_1=0.6, beta_2=0.999, epsilon=1e-8))
history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=0)
plt.plot(history.history['loss'])
print('ftml end')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['rmsprop', 'adadelta', 'adam', 'ftml'], loc='upper right')
plt.show()
