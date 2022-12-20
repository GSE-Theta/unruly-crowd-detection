from keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Input, add
from keras.models import Model, Sequential
from keras.applications.xception import preprocess_input as xception_preprocess_input

def preprocess_input(x, y):
    return xception_preprocess_input(x), y

def get_model():
    input_shape = (214, 320, 3)
    channel_axis = -1
    compressed_rate = 1 / 16

    input_layer = Input(shape=input_shape)
    x = Conv2D(32 * compressed_rate, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input_layer)
    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64 * compressed_rate, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128 * compressed_rate, (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = SeparableConv2D(128 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256 * compressed_rate, (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728 * compressed_rate, (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024 * compressed_rate, (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048 * compressed_rate, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    compressed_xception = Model(input_layer, x, name='compressed_xception')

    model = Sequential([
        compressed_xception,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(2)
    ])
    
    return model
