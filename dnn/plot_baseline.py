import numpy as np
import matplotlib.pyplot as plt
import math, os
import h5py
import tensorflow as tf
import joblib
import argparse
import matplotlib
matplotlib.use('agg')
from functions import make_mse_loss_numpy

def return_total_loss(loss, X, qcd_pred, bsm_t, bsm_pred):
    
    total_loss = []
    total_loss.append(loss(X, qcd_pred.astype(np.float32)))
    for i, bsm_i in enumerate(bsm_t):
        total_loss.append(loss(bsm_i, bsm_pred[i].astype(np.float32)))
    return total_loss

def make_feature_plots(true, prediction, xlabel, particle, bins, density, outputdir = './', ranges=None):
    print(find_min_max_range(true, prediction))
    plt.figure(figsize=(7,5))
    if ranges == None: ranges = find_min_max_range(true, prediction) 
    plt.hist(prediction, bins=bins, histtype='step', density=density, range = ranges)
    plt.hist(true, bins=bins, histtype='step', density=density, range = ranges)
    plt.yscale('log', nonpositive='clip')
    plt.ylabel('Prob. Density(a.u.)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.legend([particle+' Predicted', particle+' True'])
    if 'pT' in xlabel: xlabel = 'pT'
    elif 'phi' in xlabel: xlabel = 'phi'
    elif 'eta' in xlabel: xlabel = 'eta'
    plt.savefig(f'{outputdir}/{particle}_{xlabel}.pdf', facecolor='white')

def make_delta_feature_plots(true, prediction, xlabel, particle, bins, density, outputdir = './', ranges=None, phi=False):
    plt.figure(figsize=(7,5))
    if phi:
        delta = (true - prediction)/true
        xlabel = xlabel+' pull'
    else: 
        delta = (true - prediction)/true
        xlabel = xlabel+' pull'
    plt.hist(delta, bins=bins, histtype='step', density=density, range=ranges, label=particle)
    plt.axvline(delta.mean(), color='k', linestyle='dashed', linewidth=1, label='mean = '+str(round(delta.mean(),2)))
    plt.legend(loc='upper right')
    plt.yscale('log', nonpositive='clip')
    plt.ylabel('Prob. Density(a.u.)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.annotate('RMS =  %.2f' % np.sqrt(np.mean(delta**2)), xy=(0, 1), xytext=(12, -12), va='top',\
            xycoords='axes fraction', textcoords='offset points')
    
    if 'pT' in xlabel: xlabel = 'pT'
    elif 'phi' in xlabel: xlabel = 'phi'
    elif 'eta' in xlabel: xlabel = 'eta'
    
    plt.savefig(f'{outputdir}/{particle}_{xlabel}_zscore.pdf', facecolor='white')

def find_min_max_range(true, pred):
    minRange = min(true)
    minPred = min(pred)
    if minPred < minRange: minRange = minPred
        
    maxRange = max(true)
    maxPred = max(pred)
    if maxPred > maxRange: maxRange = maxPred
        
    return (minRange, maxRange)
            
def get_data(results):

	data = h5py.File(results, 'r')
	bsm_labels = ['VectorZPrimeToQQ__M50',
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
	              'VBF_HH']

	# QCD
	X_test_scaled = data['QCD_target'][:]
	X_test = data['QCD_input'][:]
	qcd_prediction = data['predicted_QCD'][:]

	#BSM
	bsm_prediction=[]; bsm_target = []; bsm_prediction_board=[]; bsm_data=[];bsm_prediction_onnx=[]
	for bsm in bsm_labels:
	    bsm_data.append(data[bsm+'_input'][:])
	    bsm_target.append(data[bsm+'_scaled'][:])
	    bsm_prediction.append(data['predicted_'+bsm][:])

	loss = data['loss'][:]
	val_loss = data['val_loss'][:]
	
	data.close()
	
	return loss, val_loss, bsm_data, bsm_target, bsm_prediction, X_test_scaled, X_test, qcd_prediction
    	      
def make_all_plots(results,output_dir):

	loss, val_loss, bsm_data, bsm_target, bsm_prediction, X_test_scaled, X_test, qcd_prediction = get_data(results)
	
	if not os.path.isdir(output_dir):
	 os.mkdir(output_dir) 
	
	######### plot history
	plt.figure(figsize=(10,6))
	plt.plot(loss[:], label='Training loss')
	plt.plot(val_loss[:], label='Validation loss')
	plt.title('AE - Training and validation loss')
	plt.legend(loc='best')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig(f'{output_dir}/history.pdf', facecolor='white')

        ######### mask zeros
	mask_met_delete = np.where(X_test[:,0:1].reshape(X_test.shape[0]*1)==0)[0]
	mask_eg_delete = np.where(X_test[:,1:5].reshape(X_test.shape[0]*4)==0)[0]
	mask_muon_delete = np.where(X_test[:,5:9].reshape(X_test.shape[0]*4)==0)[0]
	mask_jet_delete = np.where(X_test[:,9:19].reshape(X_test.shape[0]*10)==0)[0]

	######## reshape Test and Prediction datasets
	X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 19, 3, 1)
	qcd_pred_reshaped = qcd_prediction.reshape(qcd_prediction.shape[0], 19, 3, 1)

        ######## plot features
	# MET
	make_feature_plots(np.delete(X_test_reshaped[:,0:1,0].reshape(X_test.shape[0]*1),mask_met_delete),\
	                   np.delete(qcd_pred_reshaped[:,0:1,0].reshape(qcd_prediction.shape[0]*1),mask_met_delete),\
	                   'pT', 'MET', 100, True, output_dir)
	make_feature_plots(np.delete(X_test_reshaped[:,0:1,2].reshape(X_test_scaled.shape[0]*1),mask_met_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,0:1,2].reshape(qcd_prediction.shape[0]*1)))),mask_met_delete),\
	                   '$\phi$', 'MET', 100, True, output_dir)
	# Jets
	make_feature_plots(np.delete(X_test_reshaped[:,9:19,0].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                   np.delete(qcd_pred_reshaped[:, 9:19,0].reshape(qcd_prediction.shape[0]*10),mask_jet_delete),\
	                   'pT', 'Jets', 100, True, output_dir, ranges=(0,1000))
	make_feature_plots(np.delete(X_test_reshaped[:,9:19,1].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                np.delete(tf.make_ndarray(tf.make_tensor_proto(4.0*tf.math.tanh(qcd_pred_reshaped[:,9:19,1].reshape(qcd_prediction.shape[0]*10)))),mask_jet_delete),\
	                   '$\eta$', 'Jets', 100, True, output_dir)
	make_feature_plots(np.delete(X_test_reshaped[:,9:19,2].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,9:19,2].reshape(qcd_prediction.shape[0]*10)))),mask_jet_delete),\
	                   '$\phi$', 'Jets', 100, True, output_dir) # wrap phi
	# Muons
	make_feature_plots(np.delete(X_test_reshaped[:,5:9,0].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                   np.delete(qcd_pred_reshaped[:,5:9,0].reshape(qcd_prediction.shape[0]*4),mask_muon_delete),\
	                    'pT', 'Muons', 100, True, output_dir)
	make_feature_plots(np.delete(X_test_reshaped[:,5:9,1].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(2.1*tf.math.tanh(qcd_pred_reshaped[:,5:9,1].reshape(qcd_prediction.shape[0]*4)))),mask_muon_delete),\
	                   '$\eta$', 'Muons', 100, True, output_dir)
	make_feature_plots(np.delete(X_test_reshaped[:,5:9,2].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,5:9,2].reshape(qcd_prediction.shape[0]*4)))),mask_muon_delete),\
	                   '$\phi$', 'Muons', 100, True, output_dir)
	#EGammas
	make_feature_plots(np.delete(X_test_reshaped[:,1:5,0].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(qcd_pred_reshaped[:,1:5,0].reshape(qcd_prediction.shape[0]*4),mask_eg_delete),\
	                   'pT', 'EGammas', 100, True, output_dir, ranges = (0.75937235, 500))
	make_feature_plots(np.delete(X_test_reshaped[:,1:5,1].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(3.0*tf.math.tanh(qcd_pred_reshaped[:,1:5,1].reshape(qcd_prediction.shape[0]*4)))),mask_eg_delete),\
	                   '$\eta$', 'EGammas', 100, True, output_dir)
	make_feature_plots(np.delete(X_test_reshaped[:,1:5,2].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,1:5,2].reshape(qcd_prediction.shape[0]*4)))),mask_eg_delete),\
	                   '$\phi$', 'EGammas', 100, True, output_dir)
		   	

	# MET
	make_delta_feature_plots(np.delete(X_test_reshaped[:,0:1,0].reshape(X_test.shape[0]*1),mask_met_delete),\
	                   np.delete(qcd_pred_reshaped[:,0:1,0].reshape(qcd_prediction.shape[0]*1),mask_met_delete),\
	                   'pT', 'MET', 200, True, output_dir, ranges=(-1000, 1000))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,0:1,2].reshape(X_test.shape[0]*1),mask_met_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,0:1,2].reshape(qcd_prediction.shape[0]*1)))),mask_met_delete),\
	                   '$\phi$', 'MET', 200, True, output_dir, phi=True, ranges=(-200, 200)) # wrap phi
	# Jets
	make_delta_feature_plots(np.delete(X_test_reshaped[:,9:19,0].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                   np.delete(qcd_pred_reshaped[:, 9:19,0].reshape(qcd_prediction.shape[0]*10),mask_jet_delete),\
	                   'pT', 'Jets', 200, True, ranges=(-10000, 10000))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,9:19,1].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(4.0*tf.math.tanh(qcd_pred_reshaped[:,9:19,1].reshape(qcd_prediction.shape[0]*10)))),mask_jet_delete),\
	                   '$\eta$', 'Jets', 200, True,phi=True, ranges=(-250,250))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,9:19,2].reshape(X_test.shape[0]*10),mask_jet_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,9:19,2].reshape(qcd_prediction.shape[0]*10)))),mask_jet_delete),\
	                   '$\phi$', 'Jets', 200, True, output_dir, phi=True, ranges=(-250, 250)) # wrap phi
	# Muons
	make_delta_feature_plots(np.delete(X_test_reshaped[:,5:9,0].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                   np.delete(qcd_pred_reshaped[:,5:9,0].reshape(qcd_prediction.shape[0]*4),mask_muon_delete),\
	                    'pT', 'Muons', 200, True, output_dir, ranges=(-1000,1000))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,5:9,1].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                 np.delete(tf.make_ndarray(tf.make_tensor_proto(2.1*tf.math.tanh(qcd_pred_reshaped[:,5:9,1].reshape(qcd_prediction.shape[0]*4)))),mask_muon_delete),\
	                   '$\eta$', 'Muons', 200, True, output_dir, phi=True, ranges=(-100, 100))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,5:9,2].reshape(X_test.shape[0]*4),mask_muon_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,5:9,2].reshape(qcd_prediction.shape[0]*4)))),mask_muon_delete),\
	                  '$\phi$', 'Muons', 200, True, output_dir, phi=True, ranges=(-100, 100))
	#EGammas
	make_delta_feature_plots(np.delete(X_test_reshaped[:,1:5,0].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(qcd_pred_reshaped[:,1:5,0].reshape(qcd_prediction.shape[0]*4),mask_eg_delete),\
	                   'pT', 'EGammas', 200, True, output_dir, ranges=(-1000, 1000))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,1:5,1].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(3.0*tf.math.tanh(qcd_pred_reshaped[:,1:5,1].reshape(qcd_prediction.shape[0]*4)))),mask_eg_delete),\
	                   '$\eta$', 'EGammas', 200, True, output_dir, phi=True, ranges=(-100, 100))
	make_delta_feature_plots(np.delete(X_test_reshaped[:,1:5,2].reshape(X_test.shape[0]*4),mask_eg_delete),\
	                   np.delete(tf.make_ndarray(tf.make_tensor_proto(math.pi*tf.math.tanh(qcd_pred_reshaped[:,1:5,2].reshape(qcd_prediction.shape[0]*4)))),mask_eg_delete),\
	                   '$\phi$', 'EGammas', 200, True, output_dir, phi=True, ranges=(-100, 100))
			  	
	
	########loss distributions
	loss = make_mse_loss_numpy
	total_loss = return_total_loss(loss, X_test_scaled, qcd_prediction, bsm_target, bsm_prediction)
	
	labels = ['QCD multijet', 
	          'VectorZPrimeToQQ__M50',
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
	          'VBF_HH'] 

	minScore = 999999.
	maxScore = 0
	for i in range(len(labels)):
	    thisMin = np.min(total_loss[i])
	    thisMax = np.max(total_loss[i])
	    minScore = min(thisMin, minScore)
	    maxScore = max(maxScore, thisMax)

	bin_size=100
	plt.figure(figsize=(10,8))
	for i, label in enumerate(labels):
	    plt.hist(total_loss[i], bins=bin_size, label=label, density = True,
	         histtype='step', fill=False, linewidth=1.5, range=(minScore, 10000))
	plt.yscale('log')
	plt.xlabel("Autoencoder Loss")
	plt.ylabel("Probability (a.u.)")
	plt.grid(True)
	plt.title('MSE split loss')
	plt.legend(loc='best')
	plt.savefig(f'{output_dir}/losses.pdf', facecolor='white')
	
	
	####### ROC curves
	labels_legend = ['VectorZPrimeToQQ__M50',
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
	                'VBF_HH']
			

	from sklearn.metrics import roc_curve, auc

	target_qcd = np.zeros(total_loss[0].shape[0])

	plt.figure(figsize=(10,8))
	for i, label in enumerate(labels):
	    if i == 0: continue
    
	    trueVal = np.concatenate((np.ones(total_loss[i].shape[0]), target_qcd))
	    predVal_loss = np.concatenate((total_loss[i], total_loss[0]))

	    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

	    auc_loss = auc(fpr_loss, tpr_loss)
	    plt.plot(fpr_loss, tpr_loss, "-", label='%s (auc = %.1f%%)'%(labels_legend[i-1],auc_loss*100.), linewidth=1.5)
	    plt.semilogx()
	    plt.semilogy()
	    plt.ylabel("True Positive Rate")
	    plt.xlabel("False Positive Rate")
	    plt.grid(True)
	    plt.legend(loc='center right')
	    plt.tight_layout()
	plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
	plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
	plt.title("ROC AE")
	plt.savefig(f'{output_dir}/ROCs.pdf')
							  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, help='File with results', default='results.h5')
    parser.add_argument('--output-dir', type=str, help='Outputdir for plots', default='./')
    args = parser.parse_args()
    make_all_plots(**vars(args))
