from keras.layers import Input, Conv2D, Dense, DepthwiseConv2D, Dropout, Activation, Layer, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers import TruncatedNormal, Constant
from LayerNormalization import LayerNormalization
import keras.backend as K
import tensorflow as tf
import numpy as np
import math


def convnext(input_shape=(224,224,3), n_classes=1000,
             depths=[3,3,9,3], widths=[96,192,384,768], drop_block=0.):

    inpt = Input(input_shape)

    # stem
    x = Conv2D(widths[0], 4, strides=4, padding='same', kernel_initializer=KERNEL_INIT,
               bias_initializer=BIAS_INIT, name='stem_conv')(inpt)
    x = LayerNormalization(epsilon=1e-6, name='stem_ln')(x)

    # stages
    dbr = np.linspace(0, drop_block, sum(depths))
    for i in range(4):
        for j in range(depths[i]):
            x = ConvNextBlock(widths[i], dbr=dbr[sum(depths[:i])+j],
                              name='stage%d.block%d' % (i,j))(x)
        # downsamp: ln + s2conv
        if i<3:
            x = LayerNormalization(epsilon=1e-6, name='downsamp_ln_%d' % (i+1))(x)
            x = Conv2D(widths[i+1], kernel_size=2, strides=2, padding='same', name='downsamp_conv_%d' % (i+1),
                       kernel_initializer=KERNEL_INIT, bias_initializer=BIAS_INIT)(x)

    # final norm
    x = GlobalAveragePooling2D()(x)   # [b,c]
    x = LayerNormalization(epsilon=1e-6, name='final_ln')(x)

    # head
    x = Dense(n_classes, use_bias=True, activation='softmax', kernel_initializer=KERNEL_INIT,
              bias_initializer=BIAS_INIT, name='head')(x)

    model = Model(inpt, x)

    return model


class ConvNextBlock(Model):
    # resblock: 7x7 dwconv(neck width) - LN - 1x1 conv(bottle width) - gelu - 1x1 conv(neck width), with id path

    def __init__(self, n_filters, dbr, residual_scale=1e-6, name=None):
        super(ConvNextBlock, self).__init__(name=name)
        self.dwconv = DepthwiseConv2D(7, strides=1, padding='same', kernel_initializer=KERNEL_INIT,
                                      bias_initializer=BIAS_INIT)  # pad 3&3
        self.norm = LayerNormalization(epsilon=1e-6)
        self.pwconv1 = Conv2D(n_filters*4, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=KERNEL_INIT, bias_initializer=BIAS_INIT)
        self.pwconv2 = Conv2D(n_filters, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=KERNEL_INIT, bias_initializer=BIAS_INIT)
        self.act = Activation(gelu)
        self.residual_scale = ResidualRescale(n_filters, residual_scale)

        self.dbr = dbr
        if dbr:
            self.drop_path = Dropout(dbr, noise_shape=(None,1,1,1))

    def call(self, x):

        # id path
        skip = x

        # residual path
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.residual_scale(x)
        if self.dbr>0:
            x = self.drop_path(x)

        x = skip + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class ResidualRescale(Layer):

    def __init__(self, n_channels, init_value=1e-6):
        super(ResidualRescale, self).__init__()
        self.n_channels = n_channels
        self.init_value = init_value

    def build(self, x):
        # add trainable variable
        self.gamma = self.add_weight(shape=(self.n_channels,),
                                     initializer=Constant(value=self.init_value),
                                     name='gamma')

    def call(self, x):
        return x * self.gamma

    def compute_output_shape(self, input_shape):
        return (self.n_channels,)


def gelu(x, approx=False):
    if approx:
        return 0.5 * x * (1 + K.tanh(K.sqrt(K.constant(2./math.pi)) * (x + 0.044715 * K.pow(x, 3))))
    else:
        return 0.5 * x * (1. + tf.math.erf(x / K.sqrt(K.constant(2.))))


KERNEL_INIT = TruncatedNormal(mean=0., stddev=0.02)
BIAS_INIT = Constant(value=0)


# model zoo
def convnext_tiny():
    model = convnext(input_shape=(224,224,3), n_classes=1000,
                     depths=[3,3,9,3], widths=[96,192,384,768], drop_block=0.1)
    return model


def convnext_small():
    model = convnext(input_shape=(224,224,3), n_classes=1000,
                     depths=[3,3,27,3], widths=[96,192,384,768], drop_block=0.4)
    return model


def convnext_base():
    model = convnext(input_shape=(224,224,3), n_classes=1000,
                     depths=[3,3,27,3], widths=[128,256,512,1024], drop_block=0.5)
    return model


def convnext_large():
    model = convnext(input_shape=(224,224,3), n_classes=1000,
                     depths=[3,3,27,3], widths=[192,384,768,1536], drop_block=0.5)
    return model


if __name__ == '__main__':

    model = convnext(input_shape=(224,224,3), n_classes=21841,
                     depths=[3,3,27,3], widths=[128,256,512,1024], drop_block=0.5)
    model.summary()


