import argparse
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def prepare_data(input_file, input_bsm, events, output_file):

    # read QCD data
    with h5py.File(input_file, 'r') as h5f:
        full_data = h5f['full_data_cyl'][:events,:,:]
	
    # fit scaler to the full data
    pt_scaler = StandardScaler()
    data_target = np.copy(full_data)
    data_target[:,:,0] = pt_scaler.fit_transform(data_target[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(full_data[:,:,0],0))
    
    # define training, test and validation datasets
    X_train, X_test, Y_train, Y_test = train_test_split(full_data, data_target, test_size=0.5, shuffle=True)
    del full_data, data_target

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1)

    bsm_data = []
    if input_bsm:
        # read BSM data
        h5f2 = h5py.File(input_bsm[0],'r')
        for bsm_data_name in ['VectorZPrimeToQQ__M50',
                     'VectorZPrimeToQQ__M100',
                     'VectorZPrimeToQQ__M200',
                     'VBF_HToInvisible_M125',
                     'VBF_HToInvisible_M125_private',
                     'ZprimeToZH_MZprime1000',
                     'ZprimeToZH_MZprime800',
                     'ZprimeToZH_MZprime600',
                     'GluGluToHHTo4B',
                     'HTo2LongLivedTo4mu_1000',
                     'HTo2LongLivedTo4mu_125_12',
                     'HTo2LongLivedTo4mu_125_25',
                     'HTo2LongLivedTo4mu_125_50',
                     'VBFHToTauTau',
                     'VBF_HH'
                     ]:
            bsm_data_type = h5f2['%s' %(bsm_data_name)][:]
            bsm_data_type = bsm_data_type.reshape(bsm_data_type.shape[0],bsm_data_type.shape[1],bsm_data_type.shape[2],1)
            bsm_data.append(bsm_data_type)
        h5f2.close()

    data = [X_train, Y_train, X_test, Y_test, bsm_data, pt_scaler]

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--input-bsm', type=str, action='append', help='Input file for generated BSM')
    parser.add_argument('--events', type=int, default=-1, help='How many events to proceed')
    parser.add_argument('--output-file', type=str, help='output file', required=True)
    args = parser.parse_args()
    prepare_data(**vars(args))
