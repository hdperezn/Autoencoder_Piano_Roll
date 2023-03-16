import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from sklearn.base import  BaseEstimator, TransformerMixin, ClassifierMixin


class UNet_Pianoroll(BaseEstimator, ClassifierMixin):
    def __init__(self, img_size, loss, num_classes=1, epochs=50, batch_size=32,
                 learning_rate=1e-3, validation_split=0.2, verbose=1, droprate=0.5, filters_list=[128, 64, 32],
                 l1_l2=0, plot_loss=True):

        self.img_size = img_size
        self.loss = loss
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.verbose = verbose
        self.droprate = droprate
        self.plot_loss = plot_loss
        self.filters_list = filters_list
        self.l1_l2 = l1_l2

    def Encoder(self, img_size):
        self.encoder_inputs = keras.Input(shape=(img_size))
        ### [First half of the network: downsampling inputs] ###
        filters = self.filters_list[::-1]

        # Entry block
        x = layers.Conv2D(filters[0], 3, strides=2, padding="same", name='conv_filter32_1')(self.encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in filters[1::]:
            x = layers.Dropout(self.droprate)(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", name='Enco_conv_filter' + str(filters) + '_2')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", name='Enco_conv_filter' + str(filters) + '_3',
                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same",
                                     name='Enco_conv_filter' + str(filters) + '_4'
                                     , kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(
                previous_block_activation)
            x = layers.add([x, residual], name='Enco_add_block_filter' + str(filters))  # Add back residual
            previous_block_activation = x  # Set aside next residual
        self.x = x
        self.previous_block_activation = previous_block_activation
        self.encoder = tf.keras.Model(self.encoder_inputs, outputs=[self.x, self.previous_block_activation],
                                      name="encoder")
        return self.encoder

    def Decoder(self, num_classes):
        input_x = keras.Input(shape=(self.x.shape[1::]))
        iput_previusBlock = keras.Input(shape=(self.previous_block_activation.shape[1::]))

        x = input_x
        previous_block_activation = iput_previusBlock

        for filters in self.filters_list:
            x = layers.Dropout(self.droprate)(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", name='Deco_conv_filter' + str(filters) + '_1')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", name='Deco_conv_filter' + str(filters) + '_2',
                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", name='Deco_conv_filter' + str(filters) + '_3',
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_l2, l2=self.l1_l2))(
                residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        self.outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

        self.decoder = tf.keras.Model(inputs=[input_x, iput_previusBlock], outputs=self.outputs, name="Decoder")
        return self.decoder

    def get_model(self, *_):
        seed = 123
        tf.random.set_seed(seed)
        np.random.seed(seed)
        keras.backend.clear_session()

        self.winitializer = tf.keras.initializers.GlorotNormal(seed=seed)
        self.binitializer = "zeros"
        # ---- call layers -----
        enco = self.Encoder(self.img_size)
        deco = self.Decoder(self.num_classes)
        # ---- def red ---------
        block, previusBlock = enco(self.encoder_inputs)
        decoder_ = deco([block, previusBlock])
        # ----- MODEL -------
        metris = [tf.keras.metrics.Recall(), tf.keras.metrics.SpecificityAtSensitivity(0.1)]

        self.model = tf.keras.Model(inputs=[self.encoder_inputs], outputs=[decoder_], name='UNet_MIDI')
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss=self.loss, optimizer=opt, metrics=metris)
        return

    def fit(self, X, Y, *_):
        self.get_model()
        self.history = self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_split=self.validation_split,
                                      # callbacks=[cp_callback],
                                      verbose=self.verbose)
        # ----- plot loss -----
        if self.plot_loss:
            self.plt_history()

    def predict(self, X, *_):
        return self.model.predict(X)

    def plt_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return