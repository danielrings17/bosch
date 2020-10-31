import tensorflow as tf

def reorg_layer(input_tensor, s):
    _,h, w, c = input_tensor.get_shape().as_list()

    channels_first = tf.keras.layers.Permute((3, 1, 2))(input_tensor)

    x = tf.keras.layers.Reshape((c, h, w//s, s))(channels_first)
    x = tf.keras.layers.Permute((1, 4, 2, 3))(x)
    x = tf.keras.layers.Reshape((c, s, h // (s**2), s, s, w // s))(x)
    x = tf.keras.layers.Permute((1, 2, 4, 3, 5, 6))(x)
    x = tf.keras.layers.Reshape((c, s, s, h//s, w//s))(x)
    x = tf.keras.layers.Permute((3, 2, 1, 4, 5))(x)
    x = tf.keras.layers.Reshape((c * s * s, h // s, w // s))(x)

    channels_last = tf.keras.layers.Permute((2, 3, 1))(x)

    return channels_last

def conv_norm_lrelu(inputs, filters, kernel_size, stride=1, training=True, **conv_kwargs):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=(stride, stride), padding='same', **conv_kwargs)(inputs)
    bn = tf.keras.layers.BatchNormalization(trainable=training)(conv)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(bn)


def cnl_cascade(inputs, conv_params, training=True, **kwargs):
    x = inputs
    for param in conv_params:
        filters, ksize, stride = param
        x = conv_norm_lrelu(x, filters, ksize, stride=stride, training=training, **kwargs)
    return x
