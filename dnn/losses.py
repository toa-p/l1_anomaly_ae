import tensorflow as tf
import math
import numpy as np
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

def mse_loss(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def make_mse_loss_numpy(inputs, outputs, beta=None):
    inputs = inputs.reshape((inputs.shape[0],19,3,1))
    outputs = outputs.reshape((outputs.shape[0],19,3,1))
    
    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,0:1,:], outputs_eta_egamma[:,1:5,:], outputs_eta_muons[:,5:9,:], outputs_eta_jets[:,9:19,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0], outputs_eta[:,:,1], outputs_phi[:,:,2]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)
    reco_loss = mse_loss(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))
    if beta: return (1-beta)*reco_loss
    else: return reco_loss

def custom_loss(true, prediction):
    
    mse_loss = tf.keras.losses.MeanSquaredError()
    # 0-1 = met(pt,phi) , 2-14 = egamma, 14-26 = muon, 26-56 = jet; (pt,eta,phi) order
    #MASK PT
    mask_met = tf.math.not_equal(true[:,0:1],0)
    mask_met = tf.cast(mask_met, tf.float32)
    mask_eg = tf.math.not_equal(true[:,1:5],0)
    mask_eg = tf.cast(mask_eg, tf.float32)
    mask_muon = tf.math.not_equal(true[:,5:9],0)
    mask_muon = tf.cast(mask_muon, tf.float32)
    mask_jet = tf.math.not_equal(true[:,9:19],0)
    mask_jet = tf.cast(mask_jet, tf.float32)

    #calculate loss for pt
    #MET
    met_pt_pred = tf.math.multiply(prediction[:,0:1],mask_met)
    #met_pt_pred = tf.where(met_pt_pred==-0., 0., met_pt_pred)
    #Jets
    jets_pt_pred = tf.math.multiply(prediction[:,9:19],mask_jet)
    #jets_pt_pred = tf.where(jets_pt_pred==-0., 0., jets_pt_pred)
    #Muons
    muons_pt_pred = tf.math.multiply(prediction[:,5:9],mask_muon)
    #muons_pt_pred = tf.where(muons_pt_pred==-0., 0., muons_pt_pred)
    #EGammas
    eg_pt_pred = tf.math.multiply(prediction[:,1:5],mask_eg)
    #eg_pt_pred = tf.where(eg_pt_pred==-0., 0., eg_pt_pred)
    
    loss = mse_loss(true[:,0:1], met_pt_pred)+\
        mse_loss(true[:,9:19], jets_pt_pred) + \
        mse_loss(true[:,5:9], muons_pt_pred) + \
       mse_loss(true[:,1:5], eg_pt_pred)
    
    #Jets
    jets_eta_pred = tf.math.multiply(4.0*(tf.math.tanh(prediction[:,27:37])),mask_jet)
    #jets_eta_pred = tf.where(jets_eta_pred==-0., 0., jets_eta_pred)
    #Muons
    muons_eta_pred = tf.math.multiply(2.1*(tf.math.tanh(prediction[:,23:27])),mask_muon)
    #muons_eta_pred = tf.where(muons_eta_pred==-0., 0., muons_eta_pred)
    #EGammas
    eg_eta_pred = tf.math.multiply(3.0*(tf.math.tanh(prediction[:,19:23])),mask_eg)
    #eg_eta_pred = tf.where(eg_eta_pred==-0., 0., eg_eta_pred)

    loss += 0 + mse_loss(true[:,27:37], jets_eta_pred) + \
        mse_loss(true[:,23:27], muons_eta_pred) + \
        mse_loss(true[:,19:23], eg_eta_pred)
    
    #calculate loss for phi
    prediction_phi_met = math.pi*tf.math.tanh(prediction[:,37:38])
    prediction_phi_eg = math.pi*tf.math.tanh(prediction[:,38:42])
    prediction_phi_muon = math.pi*tf.math.tanh(prediction[:,42:46])
    prediction_phi_jets = math.pi*tf.math.tanh(prediction[:,46:56])
    
    print(prediction_phi_met.dtype)
    #MET
    met_phi_pred = tf.math.multiply(prediction_phi_met,mask_met)
    #met_phi_pred = tf.where(met_phi_pred==-0., 0., met_phi_pred)
    #Jets
    jets_phi_pred = tf.math.multiply(prediction_phi_jets,mask_jet)
    #jets_phi_pred = tf.where(jets_phi_pred==-0., 0., jets_phi_pred)
    #EGammas
    eg_phi_pred = tf.math.multiply(prediction_phi_eg,mask_eg)
    #eg_phi_pred = tf.where(eg_phi_pred==-0., 0., eg_phi_pred)
    #Muons
    muon_phi_pred = tf.math.multiply(prediction_phi_muon,mask_muon)
    #muon_phi_pred = tf.where(muon_phi_pred==-0., 0., muon_phi_pred)

    loss += mse_loss(true[:,37:38], met_phi_pred) +\
        mse_loss(true[:,46:56], jets_phi_pred) + \
       mse_loss(true[:,42:46], muon_phi_pred) + \
       mse_loss(true[:,38:42], eg_phi_pred)
    return loss

def radius(mean, logvar):
    sigma = np.sqrt(np.exp(logvar))
    radius = mean*mean/sigma/sigma
    return np.sum(radius, axis=-1)

def kl_loss(mu, logvar, beta=None):
    kl_loss = 1 + logvar - np.square(mu) - np.exp(logvar)
    kl_loss = np.mean(kl_loss, axis=-1) # mean over latent dimensions
    kl_loss *= -0.5
    if beta!=None: return beta*kl_loss
    else: return kl_loss

def mse_split_loss(true, prediction, beta=None):
   #MASK PT
    mask_met = tf.math.not_equal(true[:,0:1],0)
    mask_met = tf.cast(mask_met, tf.float32)
    mask_eg = tf.math.not_equal(true[:,1:5],0)
    mask_eg = tf.cast(mask_eg, tf.float32)
    mask_muon = tf.math.not_equal(true[:,5:9],0)
    mask_muon = tf.cast(mask_muon, tf.float32)
    mask_jet = tf.math.not_equal(true[:,9:19],0)
    mask_jet = tf.cast(mask_jet, tf.float32)

    # PT
    met_pt_pred = tf.math.multiply(prediction[:,0:1],mask_met) #MET
    jets_pt_pred = tf.math.multiply(prediction[:,9:19],mask_jet) #Jets
    muons_pt_pred = tf.math.multiply(prediction[:,5:9],mask_muon) #Muons
    eg_pt_pred = tf.math.multiply(prediction[:,1:5],mask_eg) #EGammas
    
    # ETA
    jets_eta_pred = tf.math.multiply(4.0*(tf.math.tanh(prediction[:,27:37])),mask_jet) #Jets
    muons_eta_pred = tf.math.multiply(2.1*(tf.math.tanh(prediction[:,23:27])),mask_muon) #Muons
    eg_eta_pred = tf.math.multiply(3.0*(tf.math.tanh(prediction[:,19:23])),mask_eg) #EGammas
    
    # PHI
    met_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,37:38]),mask_met) #MET
    jets_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,46:56]),mask_jet) #Jets
    muon_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,42:46]),mask_muon) #Muons
    eg_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,38:42]),mask_eg) #EGammas
    
    y_pred = tf.concat([met_pt_pred, eg_pt_pred, muons_pt_pred, jets_pt_pred, eg_eta_pred, muons_eta_pred, jets_eta_pred,\
                       met_phi_pred, eg_phi_pred, muon_phi_pred, jets_phi_pred], axis=-1)
    		       
    loss = tf.reduce_mean(tf.math.square(true - y_pred),axis=-1)
    
    if beta!= None: return (1-beta)*loss
    else: return loss
