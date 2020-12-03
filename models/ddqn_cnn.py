from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


def get_network(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(32,
                     8,
                     strides=(4, 4),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64,
                     4,
                     strides=(2, 2),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64,
                     3,
                     strides=(1, 1),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(
        lr=0.00025, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    return model
