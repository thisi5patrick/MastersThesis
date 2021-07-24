from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')


class LstmModel:
    def __new__(cls, *args, **kwargs):
        input_data = layers.Input(name='the_input', shape=(128, 64, 1), dtype='float32')  # (None, 128, 64, 1)

        iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)  # (None,64, 32, 64)

        iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

        iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
            iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)  # (None, 32, 8, 256)

        iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
            iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

        iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)

        iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
        iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

        iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)
        iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)

        iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(iam_layers)
        iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

        return Model(inputs=input_data, outputs=iam_outputs)
