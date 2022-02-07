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
    
def build_VAE(input_shape,latent_dim):
    
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
    d_input = Input(shape=(latent_dim,), name='decoder_input')
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
    