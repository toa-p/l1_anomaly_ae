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

    #Electron Isolation
    mydata = np.array(f.get('l1Ele_Iso'))
    e_iso = mydata
    del mydata
    #Muon Isolation
    mydata = np.array(f.get('l1Muon_Iso'))
    m_iso = mydata
    del mydata
    #Muon Dxy
    mydata = np.array(f.get('l1Muon_Dxy'))
    m_dxy = mydata
    del mydata

    #L1bit
    mydata = np.array(f.get('L1bit'))
    l1bit = mydata
    del mydata
    #L1 seeds
    keys = f.keys()
    seednames = []
    seedinfo = {}
    for key in keys:
        if "L1_" in key:
            seednames.append(key)
    for seed in seednames:
        mydata = np.array(f.get(seed))
        seedinfo[seed] = mydata
        del mydata

    print('The MET data shape is: ', met_cyl.shape, met_cart.shape)
    print('The e/gamma data shape is: ', e_cyl.shape, e_cart.shape, e_iso.shape)
    print('The muon data shape is: ', m_cyl.shape, m_cart.shape, m_iso.shape, m_dxy.shape)
    print('The jet data shape is: ', j_cyl.shape, j_cart.shape)
    print('The L1bit data shape is: ', l1bit.shape)

    full_data_cyl = np.concatenate([met_cyl, e_cyl, m_cyl, j_cyl], axis=1)
    full_data_cart = np.concatenate([met_cart, e_cart, m_cart, j_cart], axis=1)
    #TODO: how best to include e_iso, m_iso, m_dxy?
    full_data_iso = np.concatenate([np.zeros([met_cyl.shape[0],met_cyl.shape[1]]), e_iso, m_iso, np.zeros([j_cyl.shape[0],j_cyl.shape[1]])], axis=1)
    full_data_dxy = np.concatenate([np.zeros([met_cyl.shape[0],met_cyl.shape[1]]), np.zeros([e_cyl.shape[0],e_cyl.shape[1]]), m_dxy, np.zeros([j_cyl.shape[0],j_cyl.shape[1]])], axis=1)
    print('Done concatenating')
    print('The full cyl data shape is: ',full_data_cyl.shape)
    print('The full cart data shape is: ',full_data_cart.shape)
    print('The full iso data shape is: ',full_data_iso.shape)
    print('The full dxy data shape is: ',full_data_dxy.shape)
    return full_data_cyl, full_data_cart, full_data_iso, full_data_dxy, l1bit, seedinfo

def preprocess(input_file, output_file):

    full_data_cyl, full_data_cart, full_data_iso, full_data_dxy, l1bit, seedinfo = read_data(input_file)

    #Save this full_data_cyl
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset('full_data_cyl', data=full_data_cyl)
    h5f.create_dataset('full_data_cart', data=full_data_cart)
    h5f.create_dataset('full_data_iso', data=full_data_iso)
    h5f.create_dataset('full_data_dxy', data=full_data_dxy)
    h5f.create_dataset('L1bit', data=l1bit)
    for seed, seed_data in seedinfo.iteritems():
        h5f.create_dataset(seed, data=seed_data)
    h5f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='Path to the input file', required=True)
    parser.add_argument('--output-file', type=str, help='Path to the input file', required=True)
    args = parser.parse_args()
    preprocess(**vars(args))
