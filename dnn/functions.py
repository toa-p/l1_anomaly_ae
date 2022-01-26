import numpy as np
import math
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
#import tensorflow_probability as tfp
from qkeras import QDense, QActivation
from custom_layers import Sampling
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import h5py
from sklearn.model_selection import train_test_split

#tf.compat.v1.enable_eager_execution()

def filter_no_leptons(data):
    is_ele = data[:,1,0] > 23
    print(is_ele)
    is_mu = data[:,5,0] > 23
    print(is_mu)
    is_lep = (is_ele+is_mu) > 0
    print(is_lep)
    data_filtered = data[is_lep]
    return data_filtered

def prepare_data(input_file, input_bsm, output_file):
    # read QCD data
    with h5py.File(input_file, 'r') as h5f:
        # remove last feature, which is the type of particle
        data = h5f['Particles'][:,:,:-1]
        np.random.shuffle(data)
        #data = data[:events,:,:]
    # remove jets eta >4 or <-4
    data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
    data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
    data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
    data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
    data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
    data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])
    n_before = data.shape[0]
    data = filter_no_leptons(data)
    print('Background before filter',n_before,'after filter',data.shape[0],\
        'cut away',(n_before-data.shape[0])/n_before*100,r'%')
    # fit scaler to the full data
    pt_scaler = StandardScaler()
    data_target = np.copy(data)
    data_target[:,:,0] = pt_scaler.fit_transform(data_target[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))
    
    data = data.reshape((data.shape[0],57))
    data_target = data_target.reshape((data_target.shape[0],57))
    # define training, test and validation datasets
    X_train, X_test, Y_train, Y_test = train_test_split(data, data_target, test_size=0.5, shuffle=True)
    del data, data_target

    # read BSM data
    bsm_data = []

    with h5py.File(input_bsm[0],'r') as h5f_leptoquarks:
        leptoquarks = np.array(h5f_leptoquarks['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        leptoquarks[:,9:19,0] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,0])
        leptoquarks[:,9:19,0] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,0])
        leptoquarks[:,9:19,1] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,1])
        leptoquarks[:,9:19,1] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,1])
        leptoquarks[:,9:19,2] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,2])
        leptoquarks[:,9:19,2] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,2])
        n_before = leptoquarks.shape[0]
        leptoquarks = filter_no_leptons(leptoquarks)
        print('Leptoquarks before filter',n_before,'after filter',leptoquarks.shape[0],\
            'cut away',(n_before-leptoquarks.shape[0])/n_before*100,r'%')
        leptoquarks = leptoquarks.reshape(leptoquarks.shape[0],leptoquarks.shape[1]*leptoquarks.shape[2])
        bsm_data.append(leptoquarks)

    with h5py.File(input_bsm[1],'r') as h5f_ato4l:
        ato4l = np.array(h5f_ato4l['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        ato4l[:,9:19,0] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,0])
        ato4l[:,9:19,0] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,0])
        ato4l[:,9:19,1] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,1])
        ato4l[:,9:19,1] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,1])
        ato4l[:,9:19,2] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,2])
        ato4l[:,9:19,2] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,2])
        n_before = ato4l.shape[0]
        ato4l = filter_no_leptons(ato4l)
        print('Ato4l before filter',n_before,'after filter',ato4l.shape[0],\
            'cut away',(n_before-ato4l.shape[0])/n_before*100,r'%')
        ato4l = ato4l.reshape(ato4l.shape[0],ato4l.shape[1]*ato4l.shape[2])
        bsm_data.append(ato4l)

    with h5py.File(input_bsm[2],'r') as h5f_hChToTauNu:
        hChToTauNu = np.array(h5f_hChToTauNu['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        hChToTauNu[:,9:19,0] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,0])
        hChToTauNu[:,9:19,0] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,0])
        hChToTauNu[:,9:19,1] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,1])
        hChToTauNu[:,9:19,1] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,1])
        hChToTauNu[:,9:19,2] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,2])
        hChToTauNu[:,9:19,2] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,2])
        n_before = hChToTauNu.shape[0]
        hChToTauNu = filter_no_leptons(hChToTauNu)
        print('hChToTauNu before filter',n_before,'after filter',hChToTauNu.shape[0],\
            'cut away',(n_before-hChToTauNu.shape[0])/n_before*100,r'%')
        hChToTauNu = hChToTauNu.reshape(hChToTauNu.shape[0],hChToTauNu.shape[1]*hChToTauNu.shape[2])
        bsm_data.append(hChToTauNu)

    with h5py.File(input_bsm[3],'r') as h5f_hToTauTau:
        hToTauTau = np.array(h5f_hToTauTau['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        hToTauTau[:,9:19,0] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,0])
        hToTauTau[:,9:19,0] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,0])
        hToTauTau[:,9:19,1] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,1])
        hToTauTau[:,9:19,1] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,1])
        hToTauTau[:,9:19,2] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,2])
        hToTauTau[:,9:19,2] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,2])
        n_before = hToTauTau.shape[0]
        hToTauTau = filter_no_leptons(hToTauTau)
        print('hToTauTau before filter',n_before,'after filter',hToTauTau.shape[0],\
            'cut away',(n_before-hToTauTau.shape[0])/n_before*100,r'%')
        hToTauTau = hToTauTau.reshape(hToTauTau.shape[0],hToTauTau.shape[1]*hToTauTau.shape[2])
        bsm_data.append(hToTauTau)
    
    bsm_scaled_data=[]
    for bsm in bsm_data:
        bsm = bsm.reshape(bsm.shape[0],19,3,1)
        bsm = np.squeeze(bsm, axis=-1)
        bsm_data_target = np.copy(bsm)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm[:,:,0],0))
        bsm_data_target.reshape(bsm_data_target.shape[0], bsm_data_target.shape[1]*bsm_data_target.shape[2])
        bsm_scaled_data.append(bsm_data_target)
    
    data = [X_train, Y_train, X_test, Y_test, bsm_data, bsm_scaled_data, pt_scaler]

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_background_data(file):
    data = file['Particles'][:]
    data = data[:,:,:-1]
    
    # remove jets eta >4 or <-4
    data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
    data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
    data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
    data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
    data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
    data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])
    
    is_ele = data[:,1,0] > 23
    is_mu = data[:,5,0] > 23
    print(is_mu)
    print(is_ele)
    is_lep = (is_ele+is_mu) > 0
    print(is_lep)
    data = data[is_lep]
    
    data_noMET = data[:,1:,:]
    MET = data[:,0,[0,2]]

    pT = data_noMET[:,:,0]
    eta = data_noMET[:,:,1]
    phi = data_noMET[:,:,2]
    phi = np.concatenate((MET[:,1:2], phi), axis=1)

    pT = np.concatenate((MET[:,0:1],pT), axis=1) # add MET pt for scaling
    mask_pT = pT!=0
    
    pT_scaler = StandardScaler()
    pT_scaled = np.copy(pT)
    pT_scaled = pT_scaler.fit_transform(pT_scaled)
    pT_scaled = pT_scaled*mask_pT
    pT_scaled = np.where(pT_scaled == -0., 0, pT_scaled)
    
    data_notscaled = np.concatenate((MET[:,0:1], data_noMET[:,:,0], eta, phi), axis=1)
    data_scaled = np.concatenate((pT_scaled[:,0:1], pT_scaled[:,1:], eta, phi), axis=1)
    
    return data_scaled, data_notscaled, pT_scaler

def preprocess_anomaly_data(pT_scaler, anomaly_data):
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,2])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,2])
    
    is_ele = anomaly_data[:,1,0] > 23
    is_mu = anomaly_data[:,5,0] > 23
    is_lep = (is_ele+is_mu) > 0
    anomaly_data = anomaly_data[is_lep]
    
    data_noMET = anomaly_data[:,1:,:]
    MET = anomaly_data[:,0,[0,2]]

    pT = data_noMET[:,:,0]
    eta = data_noMET[:,:,1]
    phi = data_noMET[:,:,2]

    pT = np.concatenate((MET[:,0:1],pT), axis=1) # add MET pt for scaling
    mask_pT = pT!=0

    pT_scaled = np.copy(pT)
    pT_scaled = pT_scaler.transform(pT_scaled)
    pT_scaled = pT_scaled*mask_pT

    phi = np.concatenate((MET[:,1:2], phi), axis=1)

    test_scaled = np.concatenate((pT_scaled[:,0:1], pT_scaled[:,1:], eta, phi), axis=1)
    test_notscaled = np.concatenate((MET[:,0:1], data_noMET[:,:,0], eta, phi), axis=1)
    
    return test_scaled, test_notscaled

def prepare_data_forZenodo(file_name, output_name):
    
    anomaly_data = h5py.File(file_name, 'r')
    anomaly_data = anomaly_data['Particles'][:]
    np.random.shuffle(anomaly_data)
    ad_class = anomaly_data[:,:,-1]
    anomaly_data = anomaly_data[:,:,:-1]
#     print(file_name)
#     print("Range Ele eta: ("+str(min(anomaly_data[:,1:5,1].reshape(anomaly_data.shape[0]*4)))+", "+str(max(anomaly_data[:,1:5,1].reshape(anomaly_data.shape[0]*4)))+")")
    
#     print("Range Muon eta: ("+str(min(anomaly_data[:,5:9,1].reshape(anomaly_data.shape[0]*4)))+", "+str(max(anomaly_data[:,5:9,1].reshape(anomaly_data.shape[0]*4)))+")")
    
#     print("Range MET eta: ("+str(min(anomaly_data[:,0:1,1].reshape(anomaly_data.shape[0]*1)))+", "+str(max(anomaly_data[:,0:1,1].reshape(anomaly_data.shape[0]*1)))+")")
#     print("-------------------------------------------     ")
    
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,2])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,2])
    
    n_before=anomaly_data.shape[0]
    
    is_ele = anomaly_data[:,1,0] > 23
    is_mu = anomaly_data[:,5,0] > 23
    is_lep = (is_ele+is_mu) > 0
    data_filtered = anomaly_data[is_lep]
    ad_class_filtered = ad_class[is_lep]
    
    #print(anomaly_data.shape)
    #print(anomaly_data_class.shape)
    anomaly_data = np.concatenate((data_filtered, np.expand_dims(ad_class_filtered,axis=-1)),axis=-1)
    print(anomaly_data.shape)
    
    print('hChToTauNu before filter',n_before,'after filter',anomaly_data.shape[0],\
        'cut away',(n_before-anomaly_data.shape[0])/n_before*100,r'%')

    f = h5py.File(output_name, 'w')
    f.create_dataset('Particles', data=anomaly_data, compression='gzip')
    f.create_dataset('Particles_Names', data=np.array([b'Pt', b'Eta', b'Phi', b'Class']), compression='gzip')
    f.create_dataset('Particles_Classes', data=np.array([b'MET_class_1', b'Four_Ele_class_2', b'Four_Mu_class_3', b'Ten_Jet_class_4']), compression='gzip')
    f.close()
    
    return

def mse_loss_tf(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs - inputs), axis=-1)

def make_mse_loss(inputs, outputs):
    # remove last dimension
    inputs = tf.reshape(inputs, (tf.shape(inputs)[0],19,3,1))
    outputs = tf.reshape(outputs, (tf.shape(outputs)[0],19,3,1))
    
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs)
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs)
    outputs_eta_muons = 2.1*tf.math.tanh(outputs)
    outputs_eta_jets = 4.0*tf.math.tanh(outputs)
    outputs_eta = tf.concat([outputs[:,0:1,:], outputs_eta_egamma[:,1:5,:], outputs_eta_muons[:,5:9,:], outputs_eta_jets[:,9:19,:]], axis=1)
    # use both tricks
    outputs = tf.concat([outputs[:,:,0], outputs_eta[:,:,1], outputs_phi[:,:,2]], axis=2)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs
    loss = mse_loss_tf(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

def mse_loss_numpy(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def make_mse_loss_numpy(inputs, outputs):
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
    reco_loss = mse_loss_numpy(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))
    return reco_loss

def custom_loss_negative(true, prediction):
    
    #mse_loss = tf.keras.losses.MeanSquaredError()
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
    return -loss

def custom_loss_training(true, prediction):
    
    #mse_loss = tf.keras.losses.MeanSquaredError()
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
    loss = tf.reduce_mean(tf.math.square(true - y_pred),axis=-1, keepdims=True)
    return loss

def mse_loss(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def custom_loss_numpy(true, prediction):
    #mse_loss = tf.keras.losses.MeanSquaredError()
    # 0-1 = met(pt,phi) , 2-14 = egamma, 14-26 = muon, 26-56 = jet; (pt,eta,phi) order
    #MASK PT
    mask_met = np.not_equal(true[:,0:1],0)
    mask_eg = np.not_equal(true[:,1:5],0)
    mask_muon = np.not_equal(true[:,5:9],0)
    mask_jet = np.not_equal(true[:,9:19],0)

    # PT
    met_pt_pred = np.multiply(prediction[:,0:1],mask_met) #MET
    jets_pt_pred = np.multiply(prediction[:,9:19],mask_jet) #Jets
    muons_pt_pred = np.multiply(prediction[:,5:9],mask_muon) #Muons
    eg_pt_pred = np.multiply(prediction[:,1:5],mask_eg) #EGammas
    
    # ETA
    jets_eta_pred = np.multiply(4.0*(np.tanh(prediction[:,27:37])),mask_jet) #Jets
    muons_eta_pred = np.multiply(2.1*(np.tanh(prediction[:,23:27])),mask_muon) #Muons
    eg_eta_pred = np.multiply(3.0*(np.tanh(prediction[:,19:23])),mask_eg) #EGammas
    
    # PHI
    met_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,37:38]),mask_met) #MET
    jets_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,46:56]),mask_jet) #Jets
    muon_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,42:46]),mask_muon) #Muons
    eg_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,38:42]),mask_eg) #EGammas
    
    y_pred = np.concatenate([met_pt_pred, eg_pt_pred, muons_pt_pred, jets_pt_pred, eg_eta_pred, muons_eta_pred, jets_eta_pred,\
                       met_phi_pred, eg_phi_pred, muon_phi_pred, jets_phi_pred], axis=-1)
    loss = mse_loss(true,y_pred)
    return loss

def roc_objective(ae, X_test, bsm_data):
    def roc_objective_val(y_true, y_pred):
        # evaluate mse term
        predicted_qcd = ae(X_test, training=False)
        mse_qcd = custom_loss_numpy(X_test, predicted_qcd.numpy())
        predicted_bsm = ae(bsm_data, training=False)
        mse_bsm = custom_loss_numpy(bsm_data, predicted_bsm.numpy())
        mse_true_val = np.concatenate((np.ones(bsm_data.shape[0]), np.zeros(X_test.shape[0])), axis=-1)
        mse_pred_val = np.concatenate((mse_bsm, mse_qcd), axis=-1)
        mse_fpr_loss, mse_tpr_loss, mse_threshold_loss = roc_curve(mse_true_val, mse_pred_val)
        mse_objective = np.interp(10**(-5), mse_fpr_loss, mse_tpr_loss)
        
        objective = mse_objective # maximize
        return objective
    return roc_objective_val

def load_model(model_name, custom_objects={'QDense': QDense, 'QActivation': QActivation}):
    name = model_name + '.json'
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    model.load_weights(model_name + '.h5')
    return model

def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_save_name + '.h5')

def radius(mean, logvar):
    sigma = np.sqrt(np.exp(logvar))
    radius = mean*mean/sigma/sigma
    return np.sum(radius, axis=-1)
