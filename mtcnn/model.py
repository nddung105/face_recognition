from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, PReLU, Flatten, Dense, Softmax
from tensorflow.keras.models import Model
import numpy as np


def PNet(input_shape=None):
    if input_shape is None:
        input_shape = (None, None, 3)

    input_ = Input(input_shape)

    # Conv2D ---- 1
    x = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(input_)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

    # Conv2D --- 2
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Conv2D --- 3
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)

    output_1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(x)
    output_1 = Softmax(axis=3)(output_1)

    output_2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1))(x)

    pnet = Model(input_, [output_2, output_1])

    return pnet


def RNet(input_shape=None):
    if input_shape is None:
        input_shape = (24, 24, 3)

    input_ = Input(input_shape)

    # Conv2D --- 1
    x = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid")(input_)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Conv2D --- 2
    x = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    # Conv2D --- 3
    x = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = PReLU()(x)

    output_1 = Dense(2)(x)
    output_1 = Softmax(axis=1)(output_1)

    output_2 = Dense(4)(x)

    rnet = Model(input_, [output_2, output_1])

    return rnet


def ONet(input_shape=None):
    if input_shape is None:
        input_shape = (48, 48, 3)

    input_ = Input(input_shape)

    # Conv2D --- 1
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(input_)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Conv2D --- 2
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    # Conv2D --- 3
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

    # Conv2D --- 4
    x = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid")(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    output_1 = Dense(2)(x)
    output_1 = Softmax(axis=1)(output_1)
    output_2 = Dense(4)(x)
    output_3 = Dense(10)(x)

    onet = Model(input_, [output_2, output_3, output_1])

    return onet


def load_model(path_weights_file):
    weights = np.load(path_weights_file, allow_pickle=True).tolist()

    # Build network 3 model
    pnet = PNet()
    rnet = RNet()
    onet = ONet()

    # set weights for 3 model
    pnet.set_weights(weights['pnet'])
    rnet.set_weights(weights['rnet'])
    onet.set_weights(weights['onet'])

    return pnet, rnet, onet
