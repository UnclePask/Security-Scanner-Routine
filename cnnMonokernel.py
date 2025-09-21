'''
Created on 13 set 2025

@author: pasquale
'''
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply

class cnnMonokernel(object):
    '''
    classdocs
    '''
    def __init__(self, num_classes, input_shape):
        '''
        Constructor
        '''
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def eCa_layer(self, x, filters, kernel_size):
        attention = Conv2D(filters, kernel_size, padding="same", activation="sigmoid")(x)
        out = Multiply()([x, attention])
        return out

    def build_cnn_mono(self):
        inputs = Input(shape=self.input_shape)

        # Primo blocco
        x = Conv2D(32, 3, padding="same")(inputs)
        x = Activation("elu")(x) #elu = se x>0 x altrimenti alpha(e^x -1) differenza con RELU = max{0, x} 
        x = BatchNormalization()(x)

        x = Conv2D(32, 3, padding="same")(x)
        x = Activation("elu")(x)
        x = BatchNormalization()(x)

        x = self.eCa_layer(x, filters=1, kernel_size=3)
        x = MaxPooling2D(pool_size=(2,1))(x) #pooling (2, 2) ci dava dimensione 0 sulle righe perché abbiamo dato unidimensionale in riga

        # Secondo blocco
        x = Conv2D(32, 3, padding="same")(x)
        x = Activation("elu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, 3, padding="same")(x)
        x = Activation("elu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, 3, padding="same")(x)
        x = Activation("elu")(x)
        x = BatchNormalization()(x)

        x = self.eCa_layer(x, filters=64, kernel_size=3)
        x = MaxPooling2D(pool_size=(2,1))(x) #come prima equivale a pool_size=2 di MaxPooling1D

        # Terzo blocco
        x = Conv2D(128, 3, padding="same")(x)
        x = Activation("elu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, 3, padding="same")(x)
        x = Activation("elu")(x)

        x = self.eCa_layer(x, filters=128, kernel_size=3)

        # Global pooling + output
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs, outputs)
        # compilazione
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
    
        return model