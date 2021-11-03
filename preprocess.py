import os
import json
import h5py
import argparse
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

def read_data(input_file):
    f = h5py.File(input_file, 'r')
    #MET
    mydata = np.array(f.get('l1Sum_cyl'))
    met_cyl = mydata
    met_cyl = np.reshape(met_cyl, (met_cyl.shape[0],1,3))
    del mydata
    #Electron
    mydata = np.array(f.get('l1Ele_cyl'))
    e_cyl = mydata
    del mydata
    #Muon
    mydata = np.array(f.get('l1Muon_cyl'))
    m_cyl = mydata
    del mydata
    #Jet
    mydata = np.array(f.get('l1Jet_cyl'))
    j_cyl = mydata
    del mydata

    #MET cart
    mydata = np.array(f.get('l1Sum_cart'))
    met_cart = mydata
    met_cart = np.reshape(met_cart, (met_cart.shape[0],1,3))
    del mydata
    #Electron cart
    mydata = np.array(f.get('l1Ele_cart'))
    e_cart = mydata
    del mydata
    #Muon cart
    mydata = np.array(f.get('l1Muon_cart'))
    m_cart = mydata
    del mydata
    #Jet cart
    mydata = np.array(f.get('l1Jet_cart'))
    j_cart = mydata
    del mydata

    print('The MET data shape is: ', met_cyl.shape, met_cart.shape)
    print('The e/gamma data shape is: ', e_cyl.shape, e_cart.shape)
    print('The muon data shape is: ', m_cyl.shape, m_cart.shape)
    print('The jet data shape is: ', j_cyl.shape, j_cart.shape)

    full_data_cyl = np.concatenate([met_cyl, e_cyl, m_cyl, j_cyl], axis=1)
    full_data_cart = np.concatenate([met_cart, e_cart, m_cart, j_cart], axis=1)
    print('Done concatenating')
    return full_data_cyl, full_data_cart

def preprocess(input_file, output_file):

    full_data_cyl, full_data_cart = read_data(input_file)

    #Save this full_data_cyl
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset('full_data_cyl', data=full_data_cyl)
    h5f.create_dataset('full_data_cart', data=full_data_cart)
    h5f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='Path to the input file', required=True)
    parser.add_argument('--output-file', type=str, help='Path to the input file', required=True)
    args = parser.parse_args()
    preprocess(**vars(args))
