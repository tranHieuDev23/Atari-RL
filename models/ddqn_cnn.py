from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


def get_network(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(16,
                     8,
                     strides=4,
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32,
                     4,
                     strides=2,
                     padding='valid',
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_space))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
