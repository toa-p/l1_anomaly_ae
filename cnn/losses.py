import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

nele = 4
nmu = 4
njet = 10
ele_off = 1
mu_off = 1+nele
jet_off = 1+nele+nmu

phi_max = math.pi
ele_eta_max = 3.0
mu_eta_max = 2.1
jet_eta_max = 4.0

nfeat = 3
    
def mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def make_mse(inputs, outputs):
    # remove last dimension
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    outputs = tf.squeeze(outputs, axis=-1)
    outputs = tf.cast(outputs, dtype=tf.float32)
    outputs_pt = outputs[:,:,0]
    # trick with phi (rescaled tanh activation function)
    outputs_phi = phi_max*tf.math.tanh(outputs[:,:,2])
    # trick with eta (rescaled tanh activation function)
    outputs_eta_met = outputs[:,0:1,1]
    outputs_eta_ele = ele_eta_max*tf.math.tanh(outputs[:,ele_off:ele_off+nele,1])
    outputs_eta_mu = mu_eta_max*tf.math.tanh(outputs[:,mu_off:mu_off+nmu,1])
    outputs_eta_jet = jet_eta_max*tf.math.tanh(outputs[:,jet_off:jet_off+njet,1])
    outputs_eta = tf.concat([outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1)
    # use both tricks
    outputs = tf.stack([outputs_pt, outputs_eta, outputs_phi], axis=-1)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, dtype=tf.float32)
    outputs = mask * outputs

    loss = mse_loss(tf.reshape(inputs, [-1, (1+nele+nmu+njet)*nfeat]), tf.reshape(outputs, [-1, (1+nele+nmu+njet)*nfeat]))
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

def make_kl(z_mean, z_log_var):
    @tf.function
    def kl_loss(inputs, outputs):
        kl =  - 0.5 * (1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl = tf.reduce_mean(kl, axis=-1) # average over the latent space
        kl = tf.reduce_mean(kl, axis=-1) # average over batch
        return kl
    return kl_loss

def make_mse_kl(z_mean, z_log_var, beta):
    kl_loss = make_kl(z_mean, z_log_var)
    # multiplying back by N because input is so sparse -> average error very small
    @tf.function
    def mse_kl_loss(inputs, outputs):
        return make_mse(inputs, outputs) + kl_loss(inputs, outputs) if beta==0 \
            else (1 - beta) * make_mse(inputs, outputs) + beta * kl_loss(inputs, outputs)
    return mse_kl_loss
