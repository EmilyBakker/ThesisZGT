
from tkinter.messagebox import NO
from turtle import shape
import numpy as np
from CustomLosses import custom_binary_entropy_loss
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Masking, Bidirectional, RepeatVector, Permute, Input, Lambda, Layer, Concatenate, Dot, TimeDistributed
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten, concatenate, Activation, LeakyReLU
from tensorflow.python.keras import layers

from tensorflow.python.keras.metrics import AUC, Precision, Recall, TrueNegatives
from tensorflow.python.keras.losses import binary_crossentropy, LossFunctionWrapper
from tensorflow.python.keras import regularizers
import tensorflow.python.keras.backend as K

from tensorflow.keras.optimizers import Adam




METRICS = {'binary': [Precision(name='precision'),
           Recall(name='recall'),
           AUC(name='auc'),
           AUC(name='pr', curve='PR')]
           }



def build_lstm(lstm_units=128,
               lstm_layers=1,
               bidirectional=True,
               loss='binary_crossentropy',
               mask_value=-10.0,
               num_features=3,
               num_targets=1,
               u_dropout=0.,
               learning_rate=1e-3,
               dense_units=None,
               dense_dropout=0.,
               verbose=True,
               compile=True) -> Model:

    pred_type = 'binary' 

    inputs = Input(shape=(None, num_features), name = 'text')
    x = Masking(mask_value=mask_value)(inputs)

    for i in range(lstm_layers):
        # Add lstm layers
        if not target_replication and i==lstm_layers-1:
            # last lstm layer
            new_layer = LSTM(lstm_units, name=f'lstm_layer_{i}', dropout=u_dropout, return_sequences=False)
        else:
            new_layer = LSTM(lstm_units, name=f'lstm_layer_{i}', dropout=u_dropout, return_sequences=True)

        if bidirectional: 
            new_layer=Bidirectional(new_layer, name=new_layer.name)

        x = new_layer(x)

    if dense_units is not None:
        for dense_size in dense_units:
            
            x = Dense(dense_size, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
            x = LeakyReLU()(x)

            if dense_size > 32:
                x = Dropout(dense_dropout)(x)

    x = layers.BatchNormalization()(x)

    final_layer = Dense(num_targets, activation='sigmoid')
    outputs = final_layer(x)
    model = Model(inputs=inputs, outputs=outputs)


    if verbose: print(model.summary(90))

    metrics = METRICS.get(pred_type)
    
    optimizer = Adam(learning_rate=learning_rate)
    if compile: model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    return model


def build_dense_model(dense_units: list,
                      num_features=3,
                      dropout=[0.4, 0.2],
                      kernel_regularizer=regularizers.l2(0.001)) -> Model:

    inputs = layers.Input(shape=(None,num_features,),name='demo')
    x = layers.Dropout(dropout[0])(inputs)
    

    for i, n_units in enumerate(dense_units):
        x = layers.Dense(n_units, kernel_regularizer=kernel_regularizer, kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Dropout(dropout[1])(x)
    
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=1e-3)
    metrics = METRICS.get('binary')
    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    return model


def add_lambda_layer(model: Model, num_features=3):
    inputs = Input(shape=(None,num_features))
    x = model(inputs)
    outputs = Lambda(lambda x: x[:,-1])(x)

    new_model = Model(inputs=inputs, outputs=outputs)

    return new_model

