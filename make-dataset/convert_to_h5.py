#!/usr/bin/python
from __future__ import print_function, division
import os
import argparse
import ROOT
import numpy as np
import h5py
import re

def getBits(infile, uGTTreePath):
    fl1uGT = infile.Get(uGTTreePath)
    aliases = fl1uGT.GetListOfAliases()
    AlgoMap = {}
    for alias in aliases:
        matchbit = re.match(r"L1uGT\.m_algoDecisionInitial\[([0-9]+)\]", alias.GetTitle())
        AlgoMap[alias.GetName()] = int(matchbit.group(1))
        #print(alias.GetName(), alias.GetTitle(), matchbit.group(1))
    return AlgoMap

def filterAlgoMap(algoMap):
    wanted_keys = []
    prescale_file_name = "Prescale_2022_v0_1_1.csv"
    with open(prescale_file_name) as prescale_file:
        for line in prescale_file:
            values = line.split(',')
            #values[4] corresponds to "2E+34"
            if values[4] == "1":
                wanted_keys.append(values[1])
    filteredAlgoMap = {}
    for seedname, bit in algoMap.iteritems():
        if seedname in wanted_keys:
            filteredAlgoMap[seedname] = bit
    return filteredAlgoMap

def convert_to_h5(input_file, output_file, tree_name, uGT_tree_name):

    cylNames = ['pT', 'eta', 'phi']
    cartNames = ['px', 'py', 'pz']
    l1Jet_cyl = np.array([])
    l1Jet_cart = np.array([])
    l1mu_cyl = np.array([])
    l1mu_cart = np.array([])
    l1mu_iso = np.array([])
    l1mu_dxy = np.array([])
    l1mu_upt = np.array([])
    l1ele_cyl = np.array([])
    l1ele_cart = np.array([])
    l1ele_iso = np.array([])
    l1sum_cyl = np.array([])
    l1sum_cart = np.array([])

    inFile = ROOT.TFile.Open(input_file, 'r')
    l1Tree = inFile.Get(tree_name)
    uGTTree = inFile.Get(uGT_tree_name)

    seeds = {}
    algo_map = filterAlgoMap(getBits(inFile, uGT_tree_name))
    for seedname, bit in algo_map.iteritems():
        seeds[seedname] = np.empty([l1Tree.GetEntries()])
    seeds["L1bit"] = np.empty([l1Tree.GetEntries()])

    for i in range(l1Tree.GetEntries()):
        l1Tree.GetEntry(i)
        uGTTree.GetEntry(i)
        evt = l1Tree.L1Upgrade
        uGTevt = uGTTree.L1uGT

        for seedname, bit in algo_map.iteritems():
            seeds[seedname][i] = uGTevt.getAlgoDecisionFinal(bit)
            seeds["L1bit"][i] = (seeds["L1bit"][i] or seeds[seedname][i]).astype(int)

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
        l1muonIso   = np.array([])
        l1muonDxy   = np.array([])
        l1muonUpt   = np.array([])
        for j in range(evt.nMuons):
            myL1muon = ROOT.TLorentzVector()
            myL1muon.SetPtEtaPhiM(evt.muonEt[j], evt.muonEta[j], evt.muonPhi[j], 0.)
            my_l1mu =  np.array([myL1muon.Pt(), myL1muon.Eta(), myL1muon.Phi(),
                                 myL1muon.Px(),myL1muon.Py(),myL1muon.Pz()])
            my_l1mu = np.reshape(my_l1mu, (1, my_l1mu.shape[0]))
            # append info to arrays
            l1muonArray = np.concatenate([l1muonArray, my_l1mu], axis=0) if l1muonArray.size else my_l1mu
            l1muonIso = np.append(l1muonIso, [evt.muonIso[j]])
            l1muonDxy = np.append(l1muonDxy, [evt.muonDxy[j]])
            l1muonUpt = np.append(l1muonUpt, [evt.muonEtUnconstrained[j]])
        missing = 4-l1muonArray.shape[0]
        if missing > 0:
            zeros = np.zeros([missing, len(cylNames)+len(cartNames)])
            zeros_1d = np.zeros(missing)
            if not l1muonArray.size:
                l1muonArray = zeros
                l1muonIso = zeros_1d
                l1muonDxy = zeros_1d
                l1muonUpt = zeros_1d
            else:
                l1muonArray = np.concatenate([l1muonArray, zeros], axis=0)
                l1muonIso = np.concatenate([l1muonIso, zeros_1d], axis=0)
                l1muonDxy = np.concatenate([l1muonDxy, zeros_1d], axis=0)
                l1muonUpt = np.concatenate([l1muonUpt, zeros_1d], axis=0)
        l1muonArray = l1muonArray[l1muonArray[:,0].argsort()]
        l1muonArray = l1muonArray[::-1]
        l1muonArray =l1muonArray[:4,:]
        my_l1mu_cyl = l1muonArray[:,:len(cylNames)]
        my_l1mu_cyl= np.reshape(my_l1mu_cyl, [1,4,len(cylNames)])
        my_l1mu_cart = l1muonArray[:,len(cylNames):]
        my_l1mu_cart = np.reshape(my_l1mu_cart, [1,4,len(cartNames)])
        l1muonIso = l1muonIso[l1muonArray[:,0].argsort()]
        l1muonIso = l1muonIso[:4]
        l1muonIso = np.reshape(l1muonIso, [1,4])
        l1muonDxy = l1muonDxy[l1muonArray[:,0].argsort()]
        l1muonDxy = l1muonDxy[:4]
        l1muonDxy = np.reshape(l1muonDxy, [1,4])
        l1muonUpt = l1muonUpt[l1muonArray[:,0].argsort()]
        l1muonUpt = l1muonUpt[:4]
        l1muonUpt = np.reshape(l1muonUpt, [1,4])
        if l1mu_cyl.shape[0] == 0:
            l1mu_cyl = my_l1mu_cyl
        else:
            l1mu_cyl = np.concatenate([l1mu_cyl, my_l1mu_cyl], axis=0)
        if l1mu_cart.shape[0]  == 0:
            l1mu_cart = my_l1mu_cart
        else:
            l1mu_cart = np.concatenate([l1mu_cart, my_l1mu_cart], axis=0)
        if l1mu_iso.shape[0] == 0:
            l1mu_iso = l1muonIso
        else:
            l1mu_iso = np.concatenate([l1mu_iso, l1muonIso], axis=0)
        if l1mu_dxy.shape[0] == 0:
            l1mu_dxy = l1muonDxy
        else:
            l1mu_dxy = np.concatenate([l1mu_dxy, l1muonDxy], axis=0)
        if l1mu_upt.shape[0] == 0:
            l1mu_upt = l1muonUpt
        else:
            l1mu_upt = np.concatenate([l1mu_upt, l1muonUpt], axis=0)

        l1eleArray = np.array([])
        l1eleIso   = np.array([])
        for j in range(evt.nEGs):
            myL1ele = ROOT.TLorentzVector()
            myL1ele.SetPtEtaPhiM(evt.egEt[j], evt.egEta[j], evt.egPhi[j], 0.)
            my_l1ele =  np.array([myL1ele.Pt(), myL1ele.Eta(), myL1ele.Phi(),
                                 myL1ele.Px(),myL1ele.Py(),myL1ele.Pz()])
            my_l1ele = np.reshape(my_l1ele, (1, my_l1ele.shape[0]))
            # append info to arrays
            l1eleArray = np.concatenate([l1eleArray, my_l1ele], axis=0) if l1eleArray.shape[0] > 0 else my_l1ele
            l1eleIso = np.append(l1eleIso, [evt.egIso[j]])
        missing = 4-l1eleArray.shape[0]
        if missing > 0:
            zeros = np.zeros([missing, len(cylNames)+len(cartNames)])
            zeros_1d = np.zeros(missing)
            if not l1eleArray.size:
                l1eleArray = zeros
                l1eleIso = zeros_1d
            else:
                l1eleArray = np.concatenate([l1eleArray, zeros], axis=0)
                l1eleIso = np.concatenate([l1eleIso, zeros_1d], axis=0)
        l1eleArray = l1eleArray[l1eleArray[:,0].argsort()]
        l1eleArray = l1eleArray[::-1]
        l1eleArray = l1eleArray[:4,:]
        my_l1ele_cyl = l1eleArray[:,:len(cylNames)]
        my_l1ele_cyl = np.reshape(my_l1ele_cyl, [1,4,len(cylNames)])
        my_l1ele_cart = l1eleArray[:,len(cylNames):]
        my_l1ele_cart= np.reshape(my_l1ele_cart, [1,4,len(cartNames)])
        l1eleIso = l1eleIso[l1eleArray[:,0].argsort()]
        l1eleIso = l1eleIso[:4]
        l1eleIso = np.reshape(l1eleIso, [1,4])
        if  l1ele_cyl.shape[0] == 0:
            l1ele_cyl = my_l1ele_cyl
        else:
            l1ele_cyl = np.concatenate([l1ele_cyl, my_l1ele_cyl], axis=0)
        if l1ele_cart.shape[0] == 0:
            l1ele_cart = my_l1ele_cart
        else:
            l1ele_cart = np.concatenate([l1ele_cart, my_l1ele_cart], axis=0)
        if l1ele_iso.shape[0] == 0:
            l1ele_iso = l1eleIso
        else:
            l1ele_iso = np.concatenate([l1ele_iso, l1eleIso], axis=0)

    inFile.Close()

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
    outFile.create_dataset('l1Sum_cyl', data = l1sum_cyl, compression='gzip')
    outFile.create_dataset('l1Sum_cart', data = l1sum_cart, compression='gzip')
    for seed, values in seeds.iteritems():
        outFile.create_dataset(seed, data = values, compression='gzip')
    outFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--tree-name', type=str, default='l1UpgradeEmuTree/L1UpgradeTree')
    parser.add_argument('--uGT-tree-name', type=str, default='l1uGTEmuTree/L1uGTTree')
    args = parser.parse_args()
    convert_to_h5(**vars(args))
