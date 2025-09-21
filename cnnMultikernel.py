'''
Created on 11 set 2025

@author: pasquale
'''
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalMaxPooling2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Dropout, Reshape, Multiply
from keras.regularizers import l2

class cnnMultikernel(object):
    '''
    classdocs
    '''
    def __init__(self, num_classes, input_shape):
        '''
        Constructor
        '''
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def sequence_execitation_block(self, x, ratio=8):
        prev_layer_dim = x.shape[-1]
        seq = GlobalAveragePooling2D()(x)
        seq = Dense(prev_layer_dim // ratio)(seq)
        seq = BatchNormalization()(seq)
        seq = Activation("swish")(seq)
        seq = Dense(prev_layer_dim, activation="sigmoid")(seq)
        seq = Reshape((1, 1, prev_layer_dim))(seq)
        x = Multiply()([x, seq]) 
        return x

    def residual_block_ultra(self, x, filters, dropout_rate, kernel_sizes=[3,5]):
        shortcut = x
        for k in kernel_sizes:
            x = Conv2D(filters, kernel_size=k, kernel_regularizer=l2(1e-5), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        # Adatta canali per la residual connection se necessario
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, kernel_size=1, kernel_regularizer=l2(1e-5), padding="same")(shortcut)
        x = Add()([x, shortcut]) 
    
        x = self.sequence_execitation_block(x)
    
        x = MaxPooling2D(pool_size=(1,1))(x)
        x = Dropout(dropout_rate)(x)
        return x

#    def build_cnn_ultra(self, input_shape, num_classes):
    def build_cnn_ultra(self):
        inputs = Input(shape=self.input_shape)
    
        # Blocchi residuali
        x = self.residual_block_ultra(inputs, dropout_rate=0.2, filters=8)
        x = self.residual_block_ultra(x, dropout_rate=0.25, filters=16)
        x = self.residual_block_ultra(x, dropout_rate=0.3, filters=32)
        x = self.residual_block_ultra(x, dropout_rate=0.35, filters=64)
    
        # Pooling globale e fully connected
        gap1d = GlobalAveragePooling2D()(x)
        gmp1d = GlobalMaxPooling2D()(x)
        x = Concatenate()([gap1d, gmp1d])
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.25)(x)
    
        outputs = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        #no adamw perchè già usiamo il regularizer l2
        model.compile(optimizer=Adam(learning_rate=1e-3), 
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
        
        return model