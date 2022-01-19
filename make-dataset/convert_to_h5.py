#!/usr/bin/python
from __future__ import print_function, division
import os
import argparse
import uproot
import awkward as ak
import numpy as np
import h5py
import re

def to_np_array(ak_array, maxN=100, pad=0):
    '''convert awkward array to regular numpy array'''
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy()

def store_objects(arrays, nentries, nobj=10, obj='jet'):
    '''store objects in zero-padded numpy arrays'''
    # in case some higher pT objects are lower in the list,
    # get up to 2*nobj, then sort, then return nobj objects
    two_nobj = 2*nobj
    l1Obj_cyl = np.zeros((nentries, two_nobj, 3))
    l1Obj_cart = np.zeros((nentries, two_nobj, 3))
    l1Obj_iso = np.zeros((nentries, two_nobj))
    l1Obj_dxy = np.zeros((nentries, two_nobj))
    l1Obj_upt = np.zeros((nentries, two_nobj))
    pt = to_np_array(arrays['{}Et'.format(obj)], maxN=two_nobj)
    eta = to_np_array(arrays['{}Eta'.format(obj)], maxN=two_nobj)
    phi = to_np_array(arrays['{}Phi'.format(obj)], maxN=two_nobj)
    l1Obj_cyl[:,:,0] = pt
    l1Obj_cyl[:,:,1] = eta
    l1Obj_cyl[:,:,2] = phi
    l1Obj_cart[:,:,0] = pt*np.cos(phi)
    l1Obj_cart[:,:,1] = pt*np.sin(phi)
    l1Obj_cart[:,:,2] = pt*np.sinh(eta)
    if obj in ['eg', 'muon']:
        l1Obj_iso = to_np_array(arrays['{}Iso'.format(obj)], maxN=two_nobj)
    if obj == 'muon':
        l1Obj_dxy = to_np_array(arrays['{}Dxy'.format(obj)], maxN=two_nobj)
        l1Obj_upt = to_np_array(arrays['{}EtUnconstrained'.format(obj)], maxN=two_nobj)

    # now sort in descending pT order if needed
    sort_indices = np.argsort(-pt, axis=1)
    check_indices = np.tile(np.arange(0, two_nobj), (pt.shape[0], 1))
    if not np.allclose(sort_indices, check_indices):
        l1Obj_cyl[:,:,0] = np.take_along_axis(l1Obj_cyl[:,:,0], sort_indices, axis=1)
        l1Obj_cyl[:,:,1] = np.take_along_axis(l1Obj_cyl[:,:,1], sort_indices, axis=1)
        l1Obj_cyl[:,:,2] = np.take_along_axis(l1Obj_cyl[:,:,2], sort_indices, axis=1)
        l1Obj_cart[:,:,0] = np.take_along_axis(l1Obj_cart[:,:,0], sort_indices, axis=1)
        l1Obj_cart[:,:,1] = np.take_along_axis(l1Obj_cart[:,:,1], sort_indices, axis=1)
        l1Obj_cart[:,:,2] = np.take_along_axis(l1Obj_cart[:,:,2], sort_indices, axis=1)
        if obj in ['eg', 'muon']:
            l1Obj_iso = np.take_along_axis(l1Obj_iso, sort_indices, axis=1)
        if obj == 'muon':
            l1Obj_dxy = np.take_along_axis(l1Obj_dxy, sort_indices, axis=1)
            l1Obj_upt = np.take_along_axis(l1Obj_upt, sort_indices, axis=1)

    return l1Obj_cyl[:,:nobj], l1Obj_cart[:,:nobj], l1Obj_iso[:,:nobj], l1Obj_dxy[:,:nobj], l1Obj_upt[:,:nobj]

def getAlgoMap(input_file, uGTTreePath):
    import ROOT
    infile = ROOT.TFile.Open(input_file)
    fl1uGT = infile.Get(uGTTreePath)
    aliases = fl1uGT.GetListOfAliases()
    titles = [alias.GetTitle() for alias in aliases]
    names = [alias.GetName() for alias in aliases]
    infile.Close()
    AlgoMap = {}
    for name, title, in zip(names, titles):
        matchbit = re.match(r"L1uGT\.m_algoDecisionInitial\[([0-9]+)\]", title)
        AlgoMap[name] = int(matchbit.group(1))
    return AlgoMap

def filterAlgoMap(algoMap):
    prescale_file_name = "Prescale_2022_v0_1_1.csv"
    # [1] corresponds to algo name
    # [4] corresponds to "2E+34"
    with open(prescale_file_name) as prescale_file:
        wanted_keys = [line.split(',')[1] for line in prescale_file if line.split(',')[4] == "1"]
    return {key: algoMap[key] for key in wanted_keys}

def convert_to_h5(input_file, output_file, tree_name, uGT_tree_name):
    inFile = uproot.open(input_file)
    l1Tree = inFile[tree_name]
    uGTTree = inFile[uGT_tree_name]
    nentries = l1Tree.num_entries

    bit_arrays = uGTTree.arrays(['m_algoDecisionInitial', 'm_algoDecisionFinal'], library='np')
    initial_bits = np.stack(bit_arrays['m_algoDecisionInitial'], axis=0)
    final_bits = np.stack(bit_arrays['m_algoDecisionFinal'], axis=0)

    algo_map = getAlgoMap(input_file, uGT_tree_name)
    algo_map = filterAlgoMap(algo_map)

    seeds = {seedname: np.empty([nentries], dtype=bool) for seedname in algo_map.keys()}
    for seedname, bit in algo_map.iteritems():
        seeds[seedname][:] = final_bits[:,bit].astype(bool)
    seeds["L1bit"] = np.logical_or.reduce([seeds[seedname] for seedname in algo_map.keys()]).astype(bool)

    njets = 10
    nmuons = 4
    nelectrons = 4
    cylNames = ['pT', 'eta', 'phi']
    cartNames = ['px', 'py', 'pz']
    # variables to retrieve
    varList = ['nSums', 'sumType', 'sumEt', 'sumPhi',
               'jetEt', 'jetEta', 'jetPhi',
               'muonEt', 'muonEta', 'muonPhi', 'muonIso', 'muonDxy', 'muonEtUnconstrained',
               'egEt', 'egEta', 'egPhi', 'egIso']

    # get awkward arrays
    arrays = l1Tree.arrays(varList)

    # sums: store the following
    # kTotalEt, kTotalEtEm, kTotalHt, kMissingEt, kMissingHt,
    # with type 0, 16, 1, 2, 3
    l1sum_cyl = np.zeros((nentries, 3))
    l1sum_cart = np.zeros((nentries, 3))
    sumEt = to_np_array(arrays['sumEt'], maxN=arrays['nSums'][0])
    sumPhi = to_np_array(arrays['sumPhi'], maxN=arrays['nSums'][0])
    sumType = to_np_array(arrays['sumType'], maxN=arrays['nSums'][0])
    # index of sum to save (MET)
    metindex = np.where(sumType == 2)
    l1sum_cyl[:,0] = sumEt[metindex] # MET_pt
    l1sum_cyl[:,2] = sumPhi[metindex] # MET_phi
    l1sum_cart[:,0] = sumEt[metindex]*np.cos(sumPhi[metindex]) # MET_px
    l1sum_cart[:,1] = sumEt[metindex]*np.sin(sumPhi[metindex]) # MET_py

    # store objects: jets, muons, electrons
    l1Jet_cyl, l1Jet_cart, _, _, _ = store_objects(arrays, nentries, nobj=njets, obj='jet')
    l1mu_cyl, l1mu_cart, l1mu_iso, l1mu_dxy, l1mu_upt = store_objects(arrays, nentries, nobj=nmuons, obj='muon')
    l1ele_cyl, l1ele_cart, l1ele_iso, _, _ = store_objects(arrays, nentries, nobj=nelectrons, obj='eg')

    outFile = h5py.File(output_file, 'w')
    outFile.create_dataset('FeatureNames_cyl', data=cylNames, compression='gzip')
    outFile.create_dataset('FeatureNames_cart', data=cartNames, compression='gzip')
    outFile.create_dataset('l1Jet_cyl', data=l1Jet_cyl, compression='gzip')
    outFile.create_dataset('l1Jet_cart', data=l1Jet_cart, compression='gzip')
    outFile.create_dataset('l1Muon_cyl', data=l1mu_cyl, compression='gzip')
    outFile.create_dataset('l1Muon_cart', data=l1mu_cart, compression='gzip')
    outFile.create_dataset('l1Muon_Iso', data=l1mu_iso, compression='gzip')
    outFile.create_dataset('l1Muon_Dxy', data=l1mu_dxy, compression='gzip')
    outFile.create_dataset('l1Muon_Upt', data=l1mu_upt, compression='gzip')
    outFile.create_dataset('l1Ele_cyl', data=l1ele_cyl, compression='gzip')
    outFile.create_dataset('l1Ele_cart', data=l1ele_cart, compression='gzip')
    outFile.create_dataset('l1Ele_Iso', data=l1ele_iso, compression='gzip')
    outFile.create_dataset('l1Sum_cyl', data=l1sum_cyl, compression='gzip')
    outFile.create_dataset('l1Sum_cart', data=l1sum_cart, compression='gzip')
    for seed, values in seeds.iteritems():
        outFile.create_dataset(seed, data=values, compression='gzip')
    outFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--tree-name', type=str, default='l1UpgradeEmuTree/L1UpgradeTree')
    parser.add_argument('--uGT-tree-name', type=str, default='l1uGTEmuTree/L1uGTTree')
    args = parser.parse_args()
    convert_to_h5(**vars(args))
