from os import path

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from .model_base import ModelBase

K.set_image_data_format('channels_last')


class GruModel(ModelBase):
    def __init__(self):
        input_data = layers.Input(name='the_input', shape=(128, 64, 1), dtype='float32')

        iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal',
                                   activation='relu')(input_data)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)
        iam_layers = layers.Dropout(0.2)(iam_layers)

        iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal',
                                   activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)
        iam_layers = layers.Dropout(0.2)(iam_layers)

        iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(0.0001), activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(0.0001), activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)
        iam_layers = layers.Dropout(0.2)(iam_layers)

        iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal',
                                   activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6', activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)
        iam_layers = layers.Dropout(0.2)(iam_layers)

        iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7',
                                   kernel_regularizer=regularizers.l2(0.0001), activation='relu')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Dropout(0.2)(iam_layers)

        iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
        iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

        gru_1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal',
                           name='gru1')(iam_layers)
        gru_1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                            name='gru1_b')(iam_layers)
        reversed_gru_1b = layers.Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = layers.add([gru_1, reversed_gru_1b])
        gru1_merged = layers.BatchNormalization()(gru1_merged)

        gru_2 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal',
                           name='gru2')(gru1_merged)
        gru_2b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                            name='gru2_b')(gru1_merged)
        reversed_gru_2b = layers.Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = layers.concatenate([gru_2, reversed_gru_2b])
        gru2_merged = layers.BatchNormalization()(gru2_merged)

        iam_outputs = layers.Dense(80, kernel_initializer='he_normal', name='dense2', activation='softmax')(gru2_merged)

        self.model = Model(inputs=input_data, outputs=iam_outputs)
        self.model.load_weights(path.join(path.dirname(__file__), 'model_weights/gru_model.h5'))
