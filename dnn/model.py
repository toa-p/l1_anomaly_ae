import numpy as np
import h5py
# import setGPU
import argparse

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import math

from datetime import datetime
from tensorboard import program
import os
import pathlib
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('agg')

import pickle
from autoencoder_classes import AE

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from neptunecontrib.monitoring.keras import NeptuneMonitor
from custom_layers import Sampling

from qkeras import QDense, QActivation, QBatchNormalization
import tensorflow_model_optimization as tfmot
tsk = tfmot.sparsity.keras


def build_AE(input_shape,latent_dim):
    
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
    # ae = AE(autoencoder)
    # ae.compile(optimizer=keras.optimizers.Adam(lr=0.00001))

    return autoencoder
    
def build_VAE(input_shape, latent_dim):
    
    #encoder
    inputArray = Input(shape=(input_shape))
    x = BatchNormalization()(inputArray)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    mu = Dense(latent_dim, name = 'latent_mu', kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    logvar = Dense(latent_dim, name = 'latent_logvar', kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)

    # Use reparameterization trick to ensure correct gradient
    z = Sampling()([mu, logvar])

    # Create encoder
    encoder = Model(inputArray, [mu, logvar, z], name='encoder')
    encoder.summary()

    #decoder
    d_input = Input(shape=(int(latent_dim),), name='decoder_input')
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(d_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    dec = Dense(input_shape, kernel_initializer=tf.keras.initializers.HeUniform(seed=42))(x)

    # Create decoder 
    decoder = Model(d_input, dec, name='decoder')
    decoder.summary()
    
    # vae = VAE(encoder, decoder)
    # vae.compile(optimizer=keras.optimizers.Adam())

    return encoder,decoder

def build_QVAE(input_shape,latent_dim,quant_size=12,integer=4,symmetric=0,pruning='pruned',batch_size=1024):

    quant_size = 12
    integer = 4
    symmetric = 0
    pruning='pruned'

    if pruning == 'pruned':
        ''' How to estimate the enc step:
                num_samples = input_train.shape[0] * (1 - validation_split)
                end_step = np.ceil(num_samples / batch_size).astype(np.int32) * pruning_epochs
                so, stop pruning at the 7th epoch
        '''
        begin_step = np.ceil((input_shape*0.8)/batch_size).astype(np.int32)*5
        end_step = np.ceil((input_shape*0.8)/batch_size).astype(np.int32)*15
        print('Begin step: ' + str(begin_step) + ', End step: ' + str(end_step))
        
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=begin_step, end_step=end_step)
        print(pruning_schedule.get_config())
    
    #encoder
    inputArray = Input(shape=(input_shape))
    x = QActivation(f'quantized_bits(16,10,0,alpha=1)')(inputArray)
    x = QBatchNormalization()(x)
    x = tsk.prune_low_magnitude(Dense(32, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),\
                                                pruning_schedule=pruning_schedule)(x) if quant_size==0\
        else tsk.prune_low_magnitude(QDense(32, kernel_initializer=tf.keras.initializers.HeNormal(seed=42),\
                kernel_quantizer='quantized_bits(' + str(quant_size) + ','+str(integer)+','+ str(symmetric) +'), alpha=1',\
                bias_quantizer='quantized_bits(' + str(quant_size) + ','+ str(integer) + ',' + str(symmetric) +', alpha=1)'),\
                                                pruning_schedule=pruning_schedule)(x)
    x = tsk.prune_low_magnitude(QBatchNormalization(), pruning_schedule=pruning_schedule)(x)
    x = tsk.prune_low_magnitude(Activation('relu'),pruning_schedule=pruning_schedule)(x) if quant_size==0\
        else tsk.prune_low_magnitude(QActivation('quantized_relu(bits=' + str(quant_size) + ')'),pruning_schedule=pruning_schedule)(x)
    x = tsk.prune_low_magnitude(Dense(16, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),\
                                            pruning_schedule=pruning_schedule)(x) if quant_size==0\
        else tsk.prune_low_magnitude(QDense(16, kernel_initializer=tf.keras.initializers.HeNormal(seed=42),\
                kernel_quantizer='quantized_bits(' + str(quant_size) + ','+str(integer)+','+ str(symmetric) +', alpha=1)',\
                bias_quantizer='quantized_bits(' + str(quant_size) + ','+ str(integer) + ',' + str(symmetric) +', alpha=1)'),\
                                    pruning_schedule=pruning_schedule)(x)
    x = tsk.prune_low_magnitude(QBatchNormalization(), pruning_schedule=pruning_schedule)(x)
    x = tsk.prune_low_magnitude(Activation('relu'),pruning_schedule=pruning_schedule)(x) if quant_size==0\
        else tsk.prune_low_magnitude(QActivation('quantized_relu(bits=' + str(quant_size) + ')'),\
                                    pruning_schedule=pruning_schedule)(x)
    mu = tsk.prune_low_magnitude(Dense(latent_dim, name = 'latent_mu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)))(x) if quant_size==0\
        else tsk.prune_low_magnitude(QDense(latent_dim, kernel_initializer=tf.keras.initializers.HeNormal(seed=42),\
                kernel_quantizer='quantized_bits(' + str(16) + ',6,'+ str(symmetric) +', alpha=1)',\
                bias_quantizer='quantized_bits(' + str(16) + ',6,'+ str(symmetric) +', alpha=1)'),\
                                    pruning_schedule=pruning_schedule)(x)
    logvar = tsk.prune_low_magnitude(Dense(latent_dim, name = 'latent_logvar', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),\
                                    pruning_schedule=pruning_schedule)(x) if quant_size==0\
        else tsk.prune_low_magnitude(QDense(latent_dim, kernel_initializer=tf.keras.initializers.HeNormal(seed=42),\
                kernel_quantizer='quantized_bits(' + str(16) + ',6,'+ str(symmetric) +', alpha=1)',\
                bias_quantizer='quantized_bits(' + str(16) + ',6,'+ str(symmetric) +', alpha=1)'),\
                                    pruning_schedule=pruning_schedule)(x)
    # Use reparameterization trick to ensure correct gradient
    z = Sampling()([mu, logvar])

    # Create encoder
    encoder = Model(inputArray, [mu, logvar, z], name='encoder')    
    encoder.summary()


    #decoder
    d_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(16, kernel_initializer=tf.keras.initializers.HeNormal(seed=42))(d_input)
    x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.3)(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer=tf.keras.initializers.HeNormal(seed=42))(x)    
    x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.3)(x)
    x = Activation('relu')(x)
    dec = Dense(input_shape, kernel_initializer=tf.keras.initializers.HeNormal(seed=42))(x)
    # Create decoder
    decoder = Model(d_input, dec, name='decoder')
    decoder.summary()

    return encoder, decoder
        