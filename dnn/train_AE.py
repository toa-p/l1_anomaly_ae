import numpy as np
import h5py
import setGPU
import argparse

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K
import math

from datetime import datetime
from tensorboard import program
import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

import pickle
from autoencoder_classes import AE

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from neptunecontrib.monitoring.keras import NeptuneMonitor

def train_AE(model_type, latent_dim, input_data, output_model_h5, 
             output_model_json, output_history, batch_size, n_epochs):
    
    with open(input_data, 'rb') as f:
       X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler = pickle.load(f)

    input_shape = X_train_flatten.shape[-1]

    #encoder
    inputArray = Input(shape=(input_shape))
    x = BatchNormalization()(inputArray)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    encoder = Dense(latent_dim, kernel_initializer=tf.keras.initializers.HeUniform())(x)
    # x = BatchNormalization()(x)
    # encoder = LeakyReLU(alpha=0.3)(x)
    #decoder
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform())(encoder)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    decoder = Dense(input_shape, kernel_initializer=tf.keras.initializers.HeUniform())(x)

    #create autoencoder
    autoencoder = Model(inputs = inputArray, outputs=decoder)
    autoencoder.summary()
    ae = AE(autoencoder)
    ae.compile(optimizer=keras.optimizers.Adam(lr=0.00001))

    callbacks=[]
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',  factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6))
    callbacks.append(TerminateOnNaN())
    callbacks.append(NeptuneMonitor())
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=10, restore_best_weights=True))
    
    print("TRAINING")
    history = ae.fit(X_train_flatten, X_train_scaled, epochs = n_epochs, batch_size = batch_size,
                     validation_split=0.2,
                     callbacks=callbacks)
      
    # save model
    model_json = autoencoder.to_json()
    with open(output_model_json, 'w') as json_file:
        json_file.write(model_json)
    autoencoder.save_weights(output_model_h5)
	
    # save training history
    with open(output_history, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)	    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='AE', choices=['AE'], help='Which model to use')
    parser.add_argument('--latent-dim', type=int, required=True, help='Latent space dimension')
    parser.add_argument('--input-data', type=str, help='Training data', required=True)
    parser.add_argument('--output-model-h5', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-model-json', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-history', type=str, help='Output file with the model training history', required=True)
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True, help='Number of epochs')
    args = parser.parse_args()
    if args.model_type == 'AE': train_AE(**vars(args))
