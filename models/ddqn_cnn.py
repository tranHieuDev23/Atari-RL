from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense


def get_network(input_shape, action_space):
    inputs = Input(shape=input_shape)
    # Convolutions on the frames on the screen
    layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = Flatten()(layer3)
    layer5 = Dense(512, activation="relu")(layer4)
    action = Dense(action_space, activation="linear")(layer5)
    return Model(inputs=inputs, outputs=action)
