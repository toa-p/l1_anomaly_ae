import argparse
import pandas as pd
import numpy as np
import math
import h5py
from sklearn.model_selection import train_test_split
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functions import prepare_data
import tensorflow as tf

def prepare_data(input_file, input_bsm, events, output_file):
    # read QCD data
    with h5py.File(input_file, 'r') as h5f:
        # remove last feature, which is the type of particle
        data = h5f['full_data_cyl'][:events,:,:]
        #np.random.shuffle(data)
        #data = data[:events,:,:]
        print(data.shape)
    
    # fit scaler to the full data
    pt_scaler = StandardScaler()
    data_target = np.copy(data)
    data_target[:,:,0] = pt_scaler.fit_transform(data_target[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))

    data = data.reshape((data.shape[0],57))
    data_target = data_target.reshape((data_target.shape[0],57))
    # define training, test and validation datasets
    X_train, X_test, Y_train, Y_test = train_test_split(data, data_target, test_size=0.5)
    del data, data_target
    
    print(X_test.shape)
    print(X_test[:,1:2])
    #print(Y_test[:,19:20])
    import sys
    sys.exit()
    
    # read BSM data
    bsm_data = []
    
    import sys
    with h5py.File(input_bsm,'r') as h5f2:
        #print(h5f2.keys())
        for key in h5f2.keys():
            if len(h5f2[key].shape) < 3: continue
            bsm_file = h5f2[key][:,:,:]
            bsm = bsm_file.reshape(bsm_file.shape[0],bsm_file.shape[1]*bsm_file.shape[2])
            bsm_data.append(bsm)
    
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
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='QCD input file', required=True)
    parser.add_argument('--input-bsm', type=str, help='Input file for generated BSM')
    parser.add_argument('--events', type=int, default=-1, help='How many events to proceed')
    parser.add_argument('--output-file', type=str, help='output file', required=True)
    args = parser.parse_args()
    prepare_data(**vars(args))
