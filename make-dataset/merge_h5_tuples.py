#!/usr/bin/python
from __future__ import print_function, division
import os
import h5py
import numpy as np
import argparse

def merge_h5_tuples(output_file, input_files, bsm):

        keys = ['l1Ele_cart', 'l1Ele_cyl', 'l1Jet_cart', 'l1Jet_cyl', 'l1Muon_cart', 'l1Muon_cyl', 'l1Sum_cyl', 'l1Sum_cart']
        data = {feature: np.array for feature in keys}

        input_files = [os.path.join(input_files[0], f) for f in os.listdir(input_files[0]) if os.path.isfile(os.path.join(input_files[0], f))] \
            if len(input_files)==1 else input_files
        for k in keys:
            data[k] = np.concatenate([h5py.File(input_file, 'r')[k] for input_file in input_files], axis=0)

        #write in data
        h5f = h5py.File(output_file, 'w')
        for feature in keys:
            h5f.create_dataset(feature, data=data[feature])
        h5f.close()

def merge_h5_tuples_bsm(output_file, input_files, bsm):

        h5f = h5py.File(output_file, 'w')
        bsm_types = ['VectorZPrimeToQQ__M50',
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
                    ]
        for bsm_type in bsm_types:
            input_file = [f for f in input_files if bsm_type in f][0]
            h5f.create_dataset(bsm_type, data=h5py.File(input_file, 'r').get('full_data_cyl'))
        h5f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, help='output file', required=True)
    parser.add_argument('--input-files', type=str, nargs='+', help='input files', required=True)
    parser.add_argument('--bsm', action='store_true')
    args = parser.parse_args()
    if not args.bsm:
        merge_h5_tuples(**vars(args))
    else:
        merge_h5_tuples_bsm(**vars(args))
