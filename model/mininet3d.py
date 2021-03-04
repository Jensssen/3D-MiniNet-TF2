# -*- coding: utf-8 -*-
"""
 
File:
    mininet3d.py

Authors: soe
Date:
    26.02.21

"""

import tensorflow as tf

# USEFUL LAYERS
l2_regularizer = tf.keras.regularizers.l2


class MiniNet3D:

    def __init__(self, num_classes, input_dim):
        self.input_dim = input_dim
        self.num_class = num_classes
        self.model = None
        # build model
        self.build(True, True)

    def downsample(self, input, n_filters_out, is_training, bn=False, use_relu=False, l2=None, name="down"):
        x = tf.keras.layers.SeparableConv2D(n_filters_out, (3, 3), strides=2, padding='same', activation=None,
                                            dilation_rate=1, use_bias=False,
                                            pointwise_regularizer=l2_regularizer(0.00004))(input)
        x = tf.keras.layers.BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)
        return x

    def upsampling(self, inputs, scale):
        return tf.compat.v1.image.resize_bilinear(inputs,
                                                  size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
                                                  align_corners=True, name="lolololololol")

    def residual_separable(self, input, n_filters, is_training, dropout=0.3, dilation=1, l2=None, name="down"):
        x = tf.keras.layers.SeparableConv2D(n_filters, (3, 3), strides=1, padding='same', activation=None,
                                            dilation_rate=dilation, use_bias=False,
                                            pointwise_regularizer=l2_regularizer(0.00004))(input)
        x = tf.keras.layers.BatchNormalization(trainable=is_training)(x)
        x = tf.keras.layers.Dropout(rate=dropout)(x)
        if input.shape[3] == x.shape[3]:
            x = tf.add(x, input)
        x = tf.nn.relu(x)
        return x

    def residual_separable_multi(self, input, n_filters, is_training, dropout=0.3, dilation=1, l2=None, name="down"):
        input_b = tf.identity(input)
        x = tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)

        d2 = tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)
        d2.dilation_rate = (dilation, dilation)
        x2 = d2(input)
        x2 = tf.keras.layers.BatchNormalization(trainable=is_training)(x2)
        x2 = tf.nn.relu(x2)

        x += x2

        x = tf.keras.layers.Conv2D(n_filters, 1, strides=1, padding='same', activation=None, dilation_rate=1,
                                   use_bias=False, kernel_regularizer=l2_regularizer(0.00004))(x)
        x = tf.keras.layers.BatchNormalization(trainable=is_training)(x)

        x = tf.keras.layers.Dropout(rate=dropout)(x)

        if input.shape[3] == x.shape[3]:
            x = tf.add(x, input_b)

        x = tf.nn.relu(x)
        return x

    def encoder_module(self, input, n_filters, is_training, dropout=0.3, dilation=[1, 1], l2=None, name="down"):
        x = self.residual_separable(input, n_filters, is_training, dropout=dropout, dilation=dilation[0], l2=l2,
                                    name=name)
        x = self.residual_separable(x, n_filters, is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)
        return x

    def encoder_module_multi(self, input, n_filters, is_training, dropout=0.3, dilation=[1, 1], l2=None, name="down"):
        x = self.residual_separable_multi(input, n_filters, is_training, dropout=dropout, dilation=dilation[0], l2=l2,
                                          name=name)
        x = self.residual_separable_multi(x, n_filters, is_training, dropout=dropout, dilation=dilation[1], l2=l2,
                                          name=name)
        return x

    # Components
    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.keras.layers.MaxPool2D(size, stride, padding='VALID')(x)

        return x

    def conv_layer(self, x, kernel, filters, train_logical, name):
        x = tf.keras.layers.Conv2D(filters, kernel, padding='SAME', name=name)(x)
        x = tf.keras.layers.BatchNormalization(trainable=train_logical)(x)
        x = tf.nn.relu(x)

        return x

    def conv_transpose_layer(self, x, kernel, depth, stride, name):
        x = tf.keras.layers.Conv2DTranspose(depth, kernel, stride=stride, scope=name)(x)

        return x

    def build(self, train_logical: bool, verbose: bool):

        '''
        Calculate Relative features
        '''

        final_input = tf.keras.layers.Input(shape=self.input_dim, name="final_input")
        x_input = tf.keras.layers.Input(shape=(64, 512, 5), name="x_input")

        x_conv = tf.keras.layers.Conv2D(192, (1, 16))(final_input)
        x_conv = tf.keras.layers.BatchNormalization(trainable=train_logical)(x_conv)
        x_conv = tf.nn.relu(x_conv)

        x = self.conv_layer(final_input, (1, 1), 24, train_logical, name="point-based_local_1")
        x = self.conv_layer(x, (1, 1), 48, train_logical, name="point-based_local_2")
        x_context = self.maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")

        x = self.conv_layer(x, (1, 1), 96, train_logical, name="point-based_local_3")
        x = self.conv_layer(x, (1, 1), 192, train_logical, name="point-based_local_4")

        x_local = self.maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")

        x_context_image = tf.reshape(x_context, (-1, 16, 128, 48))
        x_context_1 = tf.image.extract_patches(x_context_image, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                               rates=[1, 1, 1, 1], padding='SAME')
        x_context_2 = tf.image.extract_patches(x_context_image, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                               rates=[1, 2, 2, 1], padding='SAME')
        x_context_3 = tf.image.extract_patches(x_context_image, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                               rates=[1, 3, 3, 1], padding='SAME')
        x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 9, 48))
        x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 9, 48))
        x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 9, 48))
        x_context_1 = tf.reshape(x_context_1, (-1, 16 * 128, 9, 48))
        x_context_2 = tf.reshape(x_context_2, (-1, 16 * 128, 9, 48))
        x_context_3 = tf.reshape(x_context_3, (-1, 16 * 128, 9, 48))
        x_context_1 = self.conv_layer(x_context_1, (1, 1), 96, train_logical, name="point-based_context_1")
        x_context_2 = self.conv_layer(x_context_2, (1, 1), 48, train_logical, name="point-based_context_2")
        x_context_3 = self.conv_layer(x_context_3, (1, 1), 48, train_logical, name="point-based_context_3")
        x_context_1 = self.maxpool_layer(x_context_1, (1, 9), (1, 1), name="point-maxpool1_cont")
        x_context_2 = self.maxpool_layer(x_context_2, (1, 9), (1, 1), name="point-maxpool2_cont")
        x_context_3 = self.maxpool_layer(x_context_3, (1, 9), (1, 1), name="point-maxpool3_cont")
        x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 96))
        x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 48))
        x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 48))

        x_context = tf.concat([x_context_3, x_context_2, x_context_1], -1)

        x_local = tf.reshape(x_local, (-1, 16, 128, 192))
        x_conv = tf.reshape(x_conv, (-1, 16, 128, 192))

        x = tf.concat([x_conv, x_local, x_context], -1)

        atten = tf.reduce_mean(x, [1, 2], keepdims=True)
        atten = tf.keras.layers.Conv2D(576, (1, 1))(atten)
        atten = tf.sigmoid(atten)
        x = tf.multiply(x, atten)
        x = self.conv_layer(x, (1, 1), 192, train_logical, name="last")

        x_branch = self.downsample(x_input, n_filters_out=192, is_training=train_logical)

        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x2 = self.encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.downsample(x2, n_filters_out=192, is_training=train_logical)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
        x = self.encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)

        x = self.upsampling(x, 2)
        x = x + x2
        x = self.residual_separable(x, 192, train_logical, dropout=0, dilation=1)
        x = self.residual_separable(x, 192, train_logical, dropout=0, dilation=1)
        x = self.residual_separable(x, 192, train_logical, dropout=0, dilation=1)
        x = self.residual_separable(x, 192, train_logical, dropout=0, dilation=1)
        x = self.upsampling(x, 2)
        x = x + x_branch

        x = self.residual_separable(x, 96, train_logical, dropout=0, dilation=1)
        x = self.residual_separable(x, 96, train_logical, dropout=0, dilation=1)
        x = self.upsampling(x, 2)

        x = self.conv_layer(x, (3, 3), self.num_class, train_logical, 'logits')

        y = tf.identity(x, name='y')

        self.model = tf.keras.Model(inputs=[final_input, x_input], outputs=[y])

        if verbose:
            self.model.summary()
