from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import Huber
from keras.optimizers import Adam


def get_network(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(32,
                     8,
                     strides=4,
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64,
                     4,
                     strides=2,
                     padding='valid',
                     activation='relu'))
    model.add(Conv2D(64,
                     3,
                     strides=1,
                     padding='valid',
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.00025, clipnorm=1.0))
    return model
