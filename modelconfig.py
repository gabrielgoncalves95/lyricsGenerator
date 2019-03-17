import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.utils import np_utils
from keras import optimizers
from keras.optimizers import RMSprop


def defineModel(maxlen, chars):
  
    print('Build model...')
    model = Sequential()
    model.add(CuDNNLSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(CuDNNLSTM(256))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

