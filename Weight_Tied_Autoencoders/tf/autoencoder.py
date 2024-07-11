import tensorflow as tf
from tensorflow import keras


CONV_KERNEL = (3, 3)
POOL_KERNEL = (2, 2)
STRIDE_SIZE = (1, 1)
PADDING_TYPE = "same"


class DownBlock(keras.layers.Layer):

    def __init__(self, filter_in, **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        self.conv_1 = keras.layers.Conv2D(filters=filter_in,
                                          kernel_size=CONV_KERNEL,
                                          padding=PADDING_TYPE)
        self.conv_2 = keras.layers.Conv2D(filters=filter_in,
                                          kernel_size=CONV_KERNEL,
                                          padding=PADDING_TYPE)
        self.batch_1 = keras.layers.BatchNormalization()
        self.batch_2 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPooling2D(pool_size=POOL_KERNEL)

    def call(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CustomConvTranspose2d(keras.layers.Layer):

    def __init__(self, filters: int,
                 kernel_size: int = 3,
                 strides: tuple = (1, 1),
                 padding: str = "same",
                 data_format: str = "channels_last", **kwargs):
        super(CustomConvTranspose2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = [1, *strides, 1]
        self.padding = padding.upper() if isinstance(padding, str) else padding
        self.data_format = "NHWC" if data_format == "channels_last" else "NCHW"

    def call(self, inputs, weights, bias=None):
        batch_size = tf.shape(inputs)[0]
        height = inputs.shape[1] * self.strides[1]
        width = inputs.shape[2] * self.strides[2]
        output_shape = [batch_size, height, width, self.filters]
        outputs = tf.nn.conv2d_transpose(inputs,
                                         weights,
                                         output_shape,
                                         strides=self.strides,
                                         padding=self.padding,
                                         data_format=self.data_format)
        if bias is not None:
            outputs = tf.nn.bias_add(outputs,
                                     bias,
                                     data_format=self.data_format)
        return outputs


def UpperBlockCustom(ConvTranspose, filter_in):
    class UpperBlock(keras.layers.Layer):
        def __init__(self, filter_in: int, **kwargs):
            super(UpperBlock, self).__init__(**kwargs)
            self.convT = ConvTranspose(filters=filter_in,
                                       kernel_size=CONV_KERNEL,
                                       strides=STRIDE_SIZE,
                                       padding=PADDING_TYPE)

        def call(self, x, w=None):
            if w is not None:
                x = self.convT(x, w)
            else:
                x = self.convT(x)
            s = x.shape[1:3]
            x = tf.image.resize(x,
                                [s[0]*2, s[1]*2],
                                method="bilinear")
            return x

    return UpperBlock(filter_in=filter_in)


class Autoencoder(keras.models.Model):

    def __init__(self, channels: int = 1, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.down_l1 = DownBlock(filter_in=64)
        self.down_l2 = DownBlock(filter_in=128)
        self.down_l3 = DownBlock(filter_in=256)
        self.up_l3 = UpperBlockCustom(keras.layers.Conv2DTranspose, 128)
        self.up_l2 = UpperBlockCustom(keras.layers.Conv2DTranspose, 64)
        self.up_l1 = UpperBlockCustom(keras.layers.Conv2DTranspose, channels)

    def call(self, x):
        x = self.down_l1(x)
        x = self.down_l2(x)
        x = self.down_l3(x)

        x = self.up_l3(x)
        x = self.up_l2(x)
        x = self.up_l1(x)
        x = keras.activations.sigmoid(x)
        return x


class AutoencoderZeroDecoder(keras.models.Model):

    def __init__(self, channels: int = 1, **kwargs):
        super(AutoencoderZeroDecoder, self).__init__(**kwargs)
        self.down_l1 = DownBlock(filter_in=64)
        self.down_l2 = DownBlock(filter_in=128)
        self.down_l3 = DownBlock(filter_in=256)
        self.up_l3 = UpperBlockCustom(CustomConvTranspose2d, 128)
        self.up_l2 = UpperBlockCustom(CustomConvTranspose2d, 64)
        self.up_l1 = UpperBlockCustom(CustomConvTranspose2d, channels)

    def call(self, x):
        x = self.down_l1(x)
        x = self.down_l2(x)
        x = self.down_l3(x)

        w1 = self.down_l1.weights[0]
        w2 = self.down_l2.weights[0]
        w3 = self.down_l3.weights[0]

        x = self.up_l3(x, w3)
        x = self.up_l2(x, w2)
        x = self.up_l1(x, w1)
        x = keras.activations.sigmoid(x)
        return x
