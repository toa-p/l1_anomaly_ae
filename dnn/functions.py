import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
import math
from qkeras import *

def prepare_data(input_file, output_file):
    # read QCD data
    with h5py.File(input_file, 'r') as h5f:
        # remove last feature, which is the type of particle
        data = h5f['full_data_cyl'][:,:,:]
        np.random.shuffle(data)
        #data = data[:events,:,:]
    
    # fit scaler to the full data
    pt_scaler = StandardScaler()
    #data_target = np.copy(data)
    data_target[:,:,0] = pt_scaler.fit_transform(data[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))
    print(data.shape)
    data = data.reshape((data.shape[0],57))
    data_target = data_target.reshape((data_target.shape[0],57))
    # define training, test and validation datasets
    X_train, X_test, Y_train, Y_test = train_test_split(data, data_target, test_size=0.5, shuffle=True)
    del data, data_target
    
    data = [X_train, Y_train, X_test, Y_test, pt_scaler]

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_model(model_name, custom_objects=None):
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
    
def mse_loss_tf(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs - inputs), axis=-1)

def make_mse_loss(inputs, outputs):
    # remove last dimension
    inputs = tf.reshape(inputs, (tf.shape(inputs)[0],19,3,1))
    outputs = tf.reshape(outputs, (tf.shape(outputs)[0],19,3,1))
    
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs[:,:,2])
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs[:,1:5,1])
    outputs_eta_muons = 2.1*tf.math.tanh(outputs[:,5:9,1])
    outputs_eta_jets = 4.0*tf.math.tanh(outputs[:,9:19,1])
    outputs_eta = tf.concat([outputs[:,0:1,:], outputs_eta_egamma, outputs_eta_muons, outputs_eta_jets], axis=1)
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