import tensorflow as tf

input_shape = (214, 320, 3)
channel_axis = -1
compressed_rate = 0.0625

input = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(32 * compressed_rate, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(input)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
x = tf.keras.layers.Conv2D(64 * compressed_rate, (3, 3), use_bias=False, name='block1_conv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)

residual = tf.keras.layers.Conv2D(128 * compressed_rate, (1, 1),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False)(x)
residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

x = tf.keras.layers.SeparableConv2D(128 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block2_sepconv1')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
x = tf.keras.layers.SeparableConv2D(128 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block2_sepconv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

x = tf.keras.layers.MaxPooling2D((3, 3),
                        strides=(2, 2),
                        padding='same',
                        name='block2_pool')(x)
x = tf.keras.layers.add([x, residual])

residual = tf.keras.layers.Conv2D(256 * compressed_rate, (1, 1), strides=(2, 2),
                            padding='same', use_bias=False)(x)
residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

x = tf.keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
x = tf.keras.layers.SeparableConv2D(256 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block3_sepconv1')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
x = tf.keras.layers.SeparableConv2D(256 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block3_sepconv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2),
                        padding='same',
                        name='block3_pool')(x)
x = tf.keras.layers.add([x, residual])

residual = tf.keras.layers.Conv2D(728 * compressed_rate, (1, 1),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False)(x)
residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

x = tf.keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block4_sepconv1')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block4_sepconv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2),
                        padding='same',
                        name='block4_pool')(x)
x = tf.keras.layers.add([x, residual])

for i in range(4):
    residual = x
    prefix = 'block' + str(i + 5)

    x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name=prefix + '_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                    name=prefix + '_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name=prefix + '_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                    name=prefix + '_sepconv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name=prefix + '_sepconv3')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                    name=prefix + '_sepconv3_bn')(x)

    x = tf.keras.layers.add([x, residual])

residual = tf.keras.layers.Conv2D(1024 * compressed_rate, (1, 1), strides=(2, 2),
                            padding='same', use_bias=False)(x)
residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

x = tf.keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
x = tf.keras.layers.SeparableConv2D(728 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block13_sepconv1')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
x = tf.keras.layers.SeparableConv2D(1024 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block13_sepconv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

x = tf.keras.layers.MaxPooling2D((3, 3),
                        strides=(2, 2),
                        padding='same',
                        name='block13_pool')(x)
x = tf.keras.layers.add([x, residual])

x = tf.keras.layers.SeparableConv2D(1536 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block14_sepconv1')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

x = tf.keras.layers.SeparableConv2D(2048 * compressed_rate, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block14_sepconv2')(x)
x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
x = tf.keras.layers.Activation('relu', name='block14_sepconv2_act')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
compressed_xception = tf.keras.models.Model(input, x, name='compressed_xception')
