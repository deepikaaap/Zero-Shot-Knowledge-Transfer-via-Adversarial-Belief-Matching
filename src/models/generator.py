from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense
from keras.layers import LeakyReLU, Input, Reshape
from keras.layers.normalization import BatchNormalization
import tensorflow.keras.backend as K


class Generator():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim

    def build_generator_model(self):
        ip = Input(shape=(self.z_dim,))
        # model = Sequential()
        n_nodes = 256 * 8 * 8
        x = Dense(n_nodes)(ip)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((8, 8, 256))(x)
        x = BatchNormalization(axis=-1)(x)
        # upsample to 16x16 and convolve
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=-1)(x)
        # upsample to 32x32 and convolve
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=-1)(x)
        # output layer
        x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        model = Model(ip, x)

        return model


if __name__ == '__main__':
    args = type('', (), {})()
    args.batch_size = 128
    args.z_dim = 100
    print(args.batch_size)
    print(args.z_dim)
    generator_model = Generator(args)
    generator_model = generator_model.build_generator_model()

    generator_model.summary()
