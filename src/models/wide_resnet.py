import tensorflow.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class WideResNet():

    def __init__(self, kernel_initializer, gamma_initializer, dropout, epsilon, weight_decay, momentum):

        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer
        self.dropout = dropout
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum

    def expand_conv(self, init, base, k, strides=(1, 1)):
        x = Conv2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          use_bias=False)(init)

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        x = BatchNormalization(axis=channel_axis, momentum=self.momentum, epsilon=self.epsilon,
                               gamma_initializer=self.gamma_initializer,beta_initializer='zeros')(x)
        x = Activation('relu')(x)

        x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          use_bias=False)(x)

        skip = Conv2D(base * k, (1, 1), padding='same', strides=strides,
                             kernel_initializer=self.kernel_initializer,
                             kernel_regularizer=l2(self.weight_decay),
                             use_bias=False)(init)

        m = Add()([x, skip])

        return m

    def basic_block(self, input, channels, k=1):
        init = input

        channel_axis = -1

        x = BatchNormalization(axis=channel_axis, momentum=self.momentum, epsilon=self.epsilon,
                               beta_initializer='zeros',gamma_initializer=self.gamma_initializer)(input)
        x = Activation('relu')(x)
        x = Conv2D(channels * k, (3, 3), padding='same', kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          use_bias=False)(x)

        if self.dropout > 0.0: x = Dropout(self.dropout)(x)

        x = BatchNormalization(axis=channel_axis, momentum=self.momentum, epsilon=self.epsilon,
                               gamma_initializer=self.gamma_initializer,beta_initializer='zeros')(x)
        x = Activation('relu')(x)
        x = Conv2D(channels * k, (3, 3), padding='same', kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          use_bias=False)(x)

        m = Add()([init, x])
        return m

    def build_Network_Block(self, x, N, nChannels, k):
        for i in range(N - 1):
            x = self.basic_block(x, nChannels, k)

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        x = BatchNormalization(axis=channel_axis, momentum=self.momentum, epsilon=self.epsilon,
                               gamma_initializer=self.gamma_initializer,beta_initializer='zeros')(x)
        x = Activation('relu')(x)

        return x

    def build_wide_resnet(self, input_dim, nb_classes=100, d=40, k=1):
        nChannels = [16, 16, 32, 64]
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        N = int((d - 4) / 6)
        ip = Input(shape=input_dim)

        # Create initial convolution block
        x = Conv2D(16, (3, 3), padding='same', kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=l2(self.weight_decay),
                          use_bias=False)(ip)
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        x = BatchNormalization(axis=channel_axis, momentum=self.momentum, epsilon=self.epsilon,
                               gamma_initializer=self.gamma_initializer,beta_initializer='zeros')(x)
        x = Activation('relu')(x)

        # 1 Network Block
        x = self.expand_conv(x, 16, k)
        x = self.build_Network_Block(x, N, nChannels[1], k)

        # 2 Network Block
        x = self.expand_conv(x, 32, k, strides=(2, 2))
        x = self.build_Network_Block(x, N, nChannels[2], k)

        # 3 Network Block
        x = self.expand_conv(x, 64, k, strides=(2, 2))
        x = self.build_Network_Block(x, N, nChannels[3], k)

        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)

        x = Dense(nb_classes,bias_initializer='zeros', kernel_regularizer=l2(self.weight_decay))(x)

        model = Model(ip, x)

        return model


if __name__ == "__main__":
    wresnet = WideResNet('he_normal', 'uniform', 0.0, 1e-5, 0.0005, 0.1)
    # input image shape
    init = (32, 32, 3)
    wrn = wresnet.build_wide_resnet(init, nb_classes=10, d=40, k=2)
    wrn.summary()
