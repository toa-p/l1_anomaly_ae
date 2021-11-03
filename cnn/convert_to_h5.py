#!/usr/bin/python
from __future__ import print_function, division
import os
import argparse
import ROOT
import numpy as np
import h5py

def convert_to_h5(input_file, output_file, tree_name):

    cylNames = ['pT', 'eta', 'phi']
    cartNames = ['px', 'py', 'pz']
    l1Jet_cyl = np.array([])
    l1Jet_cart = np.array([])
    l1mu_cyl = np.array([])
    l1mu_cart = np.array([])
    l1ele_cyl = np.array([])
    l1ele_cart = np.array([])
    l1sum_cyl = np.array([])
    l1sum_cart = np.array([])

    inFile = ROOT.TFile(input_file, 'r')
    l1Tree = inFile.Get(tree_name)

    for i in range(l1Tree.GetEntries()):
        l1Tree.GetEntry(i)
        evt = l1Tree.L1Upgrade

        # sums: store the following
        # kTotalEt, kTotalEtEm, kTotalHt, kMissingEt, kMissingHt,
        # with type 0, 16, 1, 2, 3
        sums_cyl = np.zeros((1,3))
        sums_cart = np.zeros((1,3))
        for i in range(evt.nSums):
            if evt.sumType[i] == 2:
                met = ROOT.TLorentzVector()
                met.SetPtEtaPhiM(evt.sumEt[i], 0., evt.sumPhi[i], 0.)
                sums_cyl[0,0] = met.Pt()
                sums_cyl[0,1] = 0
                sums_cyl[0,2] = met.Phi()
                sums_cart[0,0] = met.Px()
                sums_cart[0,1] = met.Py()
                sums_cart[0,2] = 0
                l1sum_cyl = np.concatenate((l1sum_cyl, sums_cyl), axis=0) if l1sum_cyl.size else sums_cyl
                l1sum_cart = np.concatenate((l1sum_cart, sums_cart), axis=0) if l1sum_cart.size else sums_cart

        l1jetArray = np.array([])
        for j in range(evt.nJets):
            myL1Jet = ROOT.TLorentzVector()
            myL1Jet.SetPtEtaPhiM(evt.jetEt[j], evt.jetEta[j], evt.jetPhi[j], 0.)
            # prepare arrays
            my_l1Jet = np.array([myL1Jet.Pt(), myL1Jet.Eta(), myL1Jet.Phi(),
                                 myL1Jet.Px(),myL1Jet.Py(),myL1Jet.Pz()])
            my_l1Jet = np.reshape(my_l1Jet, (1,my_l1Jet.shape[0]))
            # append info to arrays
            l1jetArray = np.concatenate([l1jetArray, my_l1Jet], axis=0) if l1jetArray.size else my_l1Jet
        missing = 10 - l1jetArray.shape[0]
        if missing > 0:
            zeros = np.zeros([missing, len(cylNames)+len(cartNames)])
            if not l1jetArray.size:
                l1jetArray = zeros
            else:
                l1jetArray = np.concatenate([l1jetArray, zeros], axis=0)
        # now sort jets in descending pT order
        l1jetArray = l1jetArray[l1jetArray[:,0].argsort()]
        l1jetArray = l1jetArray[::-1]
        l1jetArray = l1jetArray[:10,:]
        my_l1Jet_cyl = l1jetArray[:,:len(cylNames)]
        my_l1Jet_cyl = np.reshape(my_l1Jet_cyl, [1,10,len(cylNames)])
        my_l1Jet_cart = l1jetArray[:,len(cylNames):]
        my_l1Jet_cart = np.reshape(my_l1Jet_cart, [1,10,len(cartNames)])
        if l1Jet_cyl.shape[0] == 0:
            l1Jet_cyl = my_l1Jet_cyl
        else:
            l1Jet_cyl = np.concatenate([l1Jet_cyl, my_l1Jet_cyl], axis=0)
        if l1Jet_cart.shape[0] == 0:
            l1Jet_cart = my_l1Jet_cart
        else:
            l1Jet_cart = np.concatenate([l1Jet_cart, my_l1Jet_cart], axis=0)

        l1muonArray = np.array([])
        for j in range(evt.nMuons):
            myL1muon = ROOT.TLorentzVector()
            myL1muon.SetPtEtaPhiM(evt.muonEt[j], evt.muonEta[j], evt.muonPhi[j], 0.)
            my_l1mu =  np.array([myL1muon.Pt(), myL1muon.Eta(), myL1muon.Phi(),
                                 myL1muon.Px(),myL1muon.Py(),myL1muon.Pz()])
            my_l1mu = np.reshape(my_l1mu, (1, my_l1mu.shape[0]))
            # append info to arrays
            l1muonArray = np.concatenate([l1muonArray, my_l1mu], axis=0) if l1muonArray.size else my_l1mu
        missing = 4-l1muonArray.shape[0]
        if missing > 0:
            zeros = np.zeros([missing, len(cylNames)+len(cartNames)])
            if not l1muonArray.size:
                l1muonArray = zeros
            else:
                l1muonArray = np.concatenate([l1muonArray, zeros], axis=0)
        l1muonArray = l1muonArray[l1muonArray[:,0].argsort()]
        l1muonArray = l1muonArray[::-1]
        l1muonArray =l1muonArray[:4,:]
        my_l1mu_cyl = l1muonArray[:,:len(cylNames)]
        my_l1mu_cyl= np.reshape(my_l1mu_cyl, [1,4,len(cylNames)])
        my_l1mu_cart = l1muonArray[:,len(cylNames):]
        my_l1mu_cart = np.reshape(my_l1mu_cart, [1,4,len(cartNames)])
        if l1mu_cyl.shape[0] == 0:
            l1mu_cyl = my_l1mu_cyl
        else:
            l1mu_cyl = np.concatenate([l1mu_cyl, my_l1mu_cyl], axis=0)
        if l1mu_cart.shape[0]  == 0:
            l1mu_cart = my_l1mu_cart
        else:
            l1mu_cart = np.concatenate([l1mu_cart, my_l1mu_cart], axis=0)

        l1eleArray = np.array([])
        for j in range(evt.nEGs):
            myL1ele = ROOT.TLorentzVector()
            myL1ele.SetPtEtaPhiM(evt.egEt[j], evt.egEta[j], evt.egPhi[j], 0.)
            my_l1ele =  np.array([myL1ele.Pt(), myL1ele.Eta(), myL1ele.Phi(),
                                 myL1ele.Px(),myL1ele.Py(),myL1ele.Pz()])
            my_l1ele = np.reshape(my_l1ele, (1, my_l1ele.shape[0]))
            # append info to arrays
            l1eleArray = np.concatenate([l1eleArray, my_l1ele], axis=0) if l1eleArray.shape[0] > 0 else my_l1ele
        missing = 4-l1eleArray.shape[0]
        if missing > 0:
            zeros = np.zeros([missing, len(cylNames)+len(cartNames)])
            if not l1eleArray.size:
                l1eleArray = zeros
            else:
                l1eleArray = np.concatenate([l1eleArray, zeros], axis=0)
        l1eleArray = l1eleArray[l1eleArray[:,0].argsort()]
        l1eleArray = l1eleArray[::-1]
        l1eleArray = l1eleArray[:4,:]
        my_l1ele_cyl = l1eleArray[:,:len(cylNames)]
        my_l1ele_cyl = np.reshape(my_l1ele_cyl, [1,4,len(cylNames)])
        my_l1ele_cart = l1eleArray[:,len(cylNames):]
        my_l1ele_cart= np.reshape(my_l1ele_cart, [1,4,len(cartNames)])
        if  l1ele_cyl.shape[0] == 0:
            l1ele_cyl = my_l1ele_cyl
        else:
            l1ele_cyl = np.concatenate([l1ele_cyl, my_l1ele_cyl], axis=0)
        if l1ele_cart.shape[0] == 0:
            l1ele_cart = my_l1ele_cart
        else:
            l1ele_cart = np.concatenate([l1ele_cart, my_l1ele_cart], axis=0)

    outFile = h5py.File(output_file, 'w')
    outFile.create_dataset('FeatureNames_cyl', data=cylNames, compression='gzip')
    outFile.create_dataset('FeatureNames_cart', data=cartNames, compression='gzip')
    outFile.create_dataset('l1Jet_cyl', data=l1Jet_cyl, compression='gzip')
    outFile.create_dataset('l1Jet_cart', data=l1Jet_cart, compression='gzip')
    outFile.create_dataset('l1Muon_cyl', data=l1mu_cyl, compression='gzip')
    outFile.create_dataset('l1Muon_cart', data=l1mu_cart, compression='gzip')
    outFile.create_dataset('l1Ele_cyl', data=l1ele_cyl, compression='gzip')
    outFile.create_dataset('l1Ele_cart', data=l1ele_cart, compression='gzip')
    outFile.create_dataset('l1Sum_cyl', data = l1sum_cyl, compression='gzip')
    outFile.create_dataset('l1Sum_cart', data = l1sum_cart, compression='gzip')
    outFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--tree-name', type=str, default='l1UpgradeEmuTree/L1UpgradeTree')
    args = parser.parse_args()
    convert_to_h5(**vars(args))
