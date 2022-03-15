import argparse
import yaml
import pandas as pd
import numpy as np
import math
import h5py
from sklearn.model_selection import train_test_split
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import sys
import gc

# import setGPU
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from datetime import datetime
from tensorboard import program
import os
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import mplhep as hep
    hep.style.use(hep.style.ROOT)
    print("Using MPL HEP for ROOT style formating")
except:
    print("Instal MPL HEP for style formating")
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#DB4437", "#4285F4", "#F4B400", "#0F9D58", "purple", "goldenrod", "peru", "coral","turquoise",'gray','navy','m','darkgreen','fuchsia','steelblue']) 
from autoencoder_classes import AE,VAE

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from neptunecontrib.monitoring.keras import NeptuneMonitor
from losses import mse_split_loss, radius, kl_loss
from functions import make_mse_loss_numpy, save_model
from sklearn.metrics import roc_curve, auc


from data_preprocessing import prepare_data
from model import build_AE, build_VAE, build_QVAE, Sampling
import tensorflow_model_optimization as tfmot
tsk = tfmot.sparsity.keras





def return_total_loss(loss, bsm_t, bsm_pred):
    total_loss = loss(bsm_t, bsm_pred.astype(np.float32))
    return total_loss

def train(input_qcd,input_bsm,data_file,outdir,events,model_type,latent_dim,batch_size,n_epochs,load_pickle,quantize):

    if(load_pickle):
        print("Loading data from file",data_file)
        if(data_file==''):
            print('Please provide input pickle files')
            return
        with open(data_file, 'rb') as f:
            X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels = pickle.load(f)
    else:
        if(input_qcd==''or input_bsm==''):
            print('Please provide input H5 files')
            return
        print("Loading and saving data...")    
        X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels = prepare_data(input_qcd, input_bsm, events, '',True)
        # print(X_train_flatten.dtype, X_train_scaled.dtype, X_test_flatten.dtype, X_test_scaled.dtype, bsm_data[0].dtype, bsm_target[0].dtype)

    if not load_pickle and data_file!='':
        with open(data_file, 'wb') as f:
            pickle.dump([X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved Pickle data to disk:",data_file)
    
    if(model_type=='AE'):
        autoencoder = build_AE(X_train_flatten.shape[-1],latent_dim)
        model = AE(autoencoder)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001))

    elif(model_type=='VAE'):
        if(quantize):
            encoder, decoder = build_QVAE(X_train_flatten.shape[-1],latent_dim)
        else:
            encoder, decoder = build_VAE(X_train_flatten.shape[-1],latent_dim)
        model = VAE(encoder, decoder)
        model.compile(optimizer=keras.optimizers.Adam())

    callbacks=[]
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',  factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6))
    callbacks.append(TerminateOnNaN())
    callbacks.append(NeptuneMonitor())
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=10, restore_best_weights=True))
    
    if(quantize):
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())


    print("Training the model")

    history = model.fit(X_train_flatten, X_train_scaled,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=callbacks)

    if(model_type=='VAE'):
    	if(quantize):
    	    final_encoder = tfmot.sparsity.keras.strip_pruning(model.encoder)
    	    final_encoder.summary()
    	    final_decoder = tfmot.sparsity.keras.strip_pruning(model.decoder)
    	    final_decoder.summary()
    	    save_model(outdir+'/model_QVAE_Encoder',final_encoder)
    	    save_model(outdir+'/model_QVAE_Decoder',final_decoder)

    	    for layer in final_encoder.layers:
    	    	if hasattr(layer, "kernel_quantizer"): print(layer.name, "kernel:", str(layer.kernel_quantizer_internal), "bias:", str(layer.bias_quantizer_internal))
    	    	elif hasattr(layer, "quantizer"): print(layer.name, "quantizer:", str(layer.quantizer))

    	    for i, w in enumerate(final_encoder.get_weights()):
    	    	print(
    		    "{} -- Total:{}, Zeros: {:.2f}%".format(
    			final_encoder.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
    		    )
    		)
    					    	 
    	else:    
    	    model.save(outdir)
    else:
    	model_json = autoencoder.to_json()
    	with open(outdir+'/model.json', 'w') as json_file:
    	    json_file.write(model_json)
    	autoencoder.save_weights(outdir+'/model.h5')
    	print("Saved model to disk")


    with open(outdir+'/history.h5', 'wb') as f:
    	pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved history to disk")
        
    # Plot training & validation loss values
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    plt.plot(hist.index.to_numpy(),hist['loss'],label='Loss')
    plt.plot(hist.index.to_numpy(),hist['val_loss'],label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outdir+'/loss.pdf')
    # plt.show()

    del X_train_flatten, X_train_scaled
    gc.collect()

def get_results(input_qcd,input_bsm,data_file,outdir,events,model_type,latent_dim):

    if not os.path.exists(data_file):
     print("The data file",data_file,'does not exist! Please rerun the training step!')
     return    
    with open(data_file, 'rb') as f:
       X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels = pickle.load(f)
    
    # load trained model]
    if(model_type=='AE'):
    	with open(outdir+"/model.json", 'r') as jsonfile: config = jsonfile.read()
    	ae = tf.keras.models.model_from_json(config)    
    	ae.load_weights(outdir+"/model.h5")
    	ae.summary()
    	model = AE(ae)
    elif(model_type=='VAE'):
    	encoder, decoder = VAE.load(outdir, custom_objects={'Sampling': Sampling})
    	encoder.summary()
    	decoder.summary()
    
    	    
    # Evaluate the model
    loss = make_mse_loss_numpy
    nevents = X_test_flatten.shape[0]
    batch_size = 1000000 #this can be smaller if one still experiences OOM issues
    nbatches = int(nevents/batch_size)+1
    print("Evaluating the model - splitting prediction computation in",nbatches,"batches")

    '''
    #previous no batching code - keep it for a bit in case of problems with batching
    if(model_type=='AE'): 
        qcd_prediction = (model.autoencoder(X_test_flatten).numpy()).astype('float16')
    elif(model_type=='VAE'):
        qcd_mean, qcd_logvar, qcd_z = encoder(X_test_flatten)
        qcd_prediction = (decoder(qcd_z).numpy()).astype('float16')
    '''	
    
    #batching predictions to avoid OOM issues
    if(model_type=='AE'): 
         if nbatches==1:
             qcd_prediction = (model.autoencoder(X_test_flatten).numpy()).astype('float16')
         else:
             qcd_prediction = np.zeros((nevents, X_test_flatten.shape[1]))
             mine,maxe = 0,batch_size           
             for b in range(1,nbatches+1):
                 print(b,mine,maxe)
                 qcd_prediction[mine:maxe,:] = (model.autoencoder(X_test_flatten[mine:maxe,:]).numpy()).astype('float16')
                 if maxe==nevents: break 
                 mine,maxe = maxe+1,maxe+batch_size		    
                 if maxe > nevents: maxe=nevents       
		 	     
    if(model_type=='VAE'):
         if nbatches == 1:
             qcd_mean, qcd_logvar, qcd_z = encoder(X_test_flatten)
             qcd_prediction = (decoder(qcd_z).numpy()).astype('float16')
         else:
             qcd_mean = np.zeros((nevents, latent_dim))
             qcd_logvar = np.zeros((nevents, latent_dim))
             qcd_z = np.zeros((nevents, latent_dim))
             qcd_prediction = np.zeros((nevents, X_test_flatten.shape[1]))	   
             mine,maxe = 0,batch_size           
             for b in range(1,nbatches+1):
                 print(b,mine,maxe)                                 
                 qcd_mean[mine:maxe,:], qcd_logvar[mine:maxe,:], qcd_z[mine:maxe,:] = encoder(X_test_flatten[mine:maxe,:])
                 qcd_prediction[mine:maxe,:] = (decoder(qcd_z[mine:maxe,:]).numpy()).astype('float16')
                 if maxe==nevents: break 
                 mine,maxe = maxe+1,maxe+batch_size		    
                 if maxe > nevents: maxe=nevents
    
    results={}
    for i, label in enumerate(bsm_labels):
        results[label] = {}
        if(model_type=='AE'): 
            bsm_pred = model.autoencoder(bsm_data[i]).numpy()
        elif(model_type=='VAE'): 
            mean_pred, logvar_pred, z_pred = encoder(bsm_data[i])
            bsm_pred = decoder(z_pred).numpy()
        results[label]['target'] = bsm_target[i]
        results[label]['prediction'] = bsm_pred
        if(model_type=='VAE'):
            results[label]['mean_prediction'] = mean_pred.numpy()
            results[label]['logvar_prediction'] = logvar_pred.numpy()
            results[label]['z_prediction'] = z_pred.numpy()
            results[label]['kl_loss'] = np.nan_to_num(kl_loss(mean_pred.numpy(),logvar_pred.numpy()))

        total_loss = return_total_loss(loss, bsm_target[i], bsm_pred)
        results[label]['loss'] = total_loss
        if(model_type=='VAE'):
            results[label]['total_loss'] = np.nan_to_num(kl_loss(mean_pred.numpy(),z_pred.numpy())+total_loss)
            results[label]['radius'] = np.nan_to_num(radius(mean_pred.numpy(),z_pred.numpy()))            

    results['QCD'] = {}
    results['QCD']['target'] = X_test_scaled
    results['QCD']['prediction'] = qcd_prediction
    qcd_loss = return_total_loss(loss, X_test_scaled, qcd_prediction)
    results['QCD']['loss'] = qcd_loss
    if(model_type=='VAE'):
        results['QCD']['mean_prediction'] = qcd_mean
        results['QCD']['logvar_prediction'] = qcd_logvar
        results['QCD']['z_prediction'] = qcd_z
        results['QCD']['kl_loss'] = np.nan_to_num(kl_loss(qcd_mean,qcd_logvar))
        results['QCD']['total_loss']=np.nan_to_num(kl_loss(qcd_mean,qcd_logvar)+qcd_loss)
        results['QCD']['radius']=np.nan_to_num(radius(qcd_mean,qcd_logvar))
	
    output_result = outdir+'/results.h5'
    h5f = h5py.File(output_result, 'w')
    h5f.create_dataset('bsm_labels', data=bsm_labels)
    h5f.create_dataset('QCD_input', data=X_test_flatten)
    h5f.create_dataset('QCD_target', data=X_test_scaled)
    h5f.create_dataset('predicted_QCD', data = qcd_prediction)
    for i, key in enumerate(results):
        if key!='QCD':
    	    h5f.create_dataset('%s_scaled' %key, data=results[key]['target'])
    	    h5f.create_dataset('%s_input' %key, data=bsm_data[i])
    	    h5f.create_dataset('predicted_%s' %key, data=results[key]['prediction'])
        h5f.create_dataset('mse_loss_%s' %key, data=results[key]['loss']) #mse only
        if(model_type=='VAE'):
    	    h5f.create_dataset('encoded_mean_%s' %key, data=results[key]['mean_prediction'])
    	    h5f.create_dataset('encoded_logvar_%s' %key, data=results[key]['logvar_prediction'])
    	    h5f.create_dataset('encoded_z_%s' %key, data=results[key]['z_prediction'])
    	    h5f.create_dataset('kl_loss_%s' %key, data=results[key]['kl_loss'])	    
    	    h5f.create_dataset('total_loss_%s' %key, data=results[key]['total_loss']) #mse+kl
    	    h5f.create_dataset('radius_%s' %key, data=results[key]['radius'])
    
    print("*** OutputFile Created")
    h5f.close()
    
    return results
    	        
def evaluate(input_qcd,input_bsm,data_file,outdir,events,model_type,latent_dim,batch_size,n_epochs,load_pickle,quantize,load_results):

    if not load_results: get_results(input_qcd,input_bsm,data_file,outdir,events,model_type,latent_dim)
    results = h5py.File(outdir+'/results.h5', 'r')    
    bsm_labels = [k.decode('utf-8') for k in results['bsm_labels'][()]]
    
    min_loss,max_loss=1e5,0
    if(model_type=='VAE'):
        min_tloss,max_tloss=1e5,0
        min_r,max_r=1e5,0
        min_kloss,max_kloss=1e5,0
    for key in bsm_labels:
        if(np.min(results['mse_loss_%s'%key])<min_loss): min_loss = np.min(results['mse_loss_%s'%key])
        if(np.mean(results['mse_loss_%s'%key])+10*np.std(results['mse_loss_%s'%key])>max_loss): max_loss = np.mean(results['mse_loss_%s'%key])+10*np.std(results['mse_loss_%s'%key])
        if(model_type=='VAE'):
            if(np.min(results['total_loss_%s'%key])<min_tloss): min_tloss = np.min(results['total_loss_%s'%key])
            if(np.mean(results['total_loss_%s'%key])+10*np.std(results['total_loss_%s'%key])>max_tloss): max_tloss = np.mean(results['total_loss_%s'%key])+10*np.std(results['total_loss_%s'%key])
            if(np.min(results['radius_%s'%key])<min_r): min_r = np.min(results['radius_%s'%key])
            if(np.mean(results['radius_%s'%key])+10*np.std(results['radius_%s'%key])>max_r): max_r = np.mean(results['radius_%s'%key])+10*np.std(results['radius_%s'%key])
            if(np.min(results['kl_loss_%s'%key])<min_kloss): min_kloss = np.min(results['kl_loss_%s'%key])
            if(np.mean(results['kl_loss_%s'%key])+10*np.std(results['kl_loss_%s'%key])>max_kloss): max_kloss = np.mean(results['kl_loss_%s'%key])+10*np.std(results['kl_loss_%s'%key])
	    	    
    # Plot the results
    print("Plotting the results")
    bins_=np.linspace(min_loss,max_loss,100)
    plt.figure(figsize=(10,10))
    plt.hist(results['mse_loss_QCD'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
    for key in bsm_labels: plt.hist(results['mse_loss_%s'%key],label=key,histtype='step',bins=bins_,density=True)
    plt.legend(fontsize='x-small')
    plt.yscale('log')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.title('Loss distribution')
    plt.savefig(outdir+'/mse_loss_hist_'+model_type+'.pdf')
    # plt.show()

    if(model_type=='VAE'):

        bins_=np.linspace(min_tloss,max_tloss,100)
        plt.figure(figsize=(10,10))
        plt.hist(results['total_loss_QCD'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
        for key in bsm_labels: plt.hist(results['total_loss_%s'%key],label=key,histtype='step',bins=bins_,density=True)
        plt.legend(fontsize='x-small')
        plt.yscale('log')
        plt.xlabel('Total Loss')
        plt.ylabel('Density')
        plt.title('Total Loss distribution')
        plt.savefig(outdir+'/total_loss_hist_'+model_type+'.pdf')

        bins_=np.linspace(min_r,max_r,100)
        plt.figure(figsize=(10,10))
        plt.hist(results['radius_QCD'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
        for key in bsm_labels: plt.hist(results['radius_%s'%key],label=key,histtype='step',bins=bins_,density=True)
        plt.legend(fontsize='x-small')
        plt.yscale('log')
        plt.xlabel('Radius')
        plt.ylabel('Density')
        plt.title('Radius distribution')
        plt.savefig(outdir+'/radius_hist_'+model_type+'.pdf')
        
        for key in bsm_labels:
            plt.figure(figsize=(10,10))
            for i in range(latent_dim):
                plt.hist(results['encoded_mean_%s'%key][:,i],bins=100,label='mean '+str(i),histtype='step', density=True,range=[-5,5])
            plt.legend(fontsize='x-small')
            plt.xlabel('Loss')
            plt.ylabel('z')
            plt.title(key+' mean Z distribution')
            plt.savefig(outdir+'/mean_z_'+model_type+'_'+key+'.pdf')
            # plt.show()

        for key in bsm_labels:
            plt.figure(figsize=(10,10))
            for i in range(latent_dim):
                plt.hist(results['encoded_logvar_%s'%key][:,i],bins=100,label='logvar '+str(i),histtype='step', density=True,range=[-20,20])
            plt.legend(fontsize='x-small')
            plt.xlabel('Loss')
            plt.ylabel('z')
            plt.title(key+' logvar Z distribution')
            plt.savefig(outdir+'/logvar_z_'+model_type+'_'+key+'.pdf')
            # plt.show()
    
    plt.figure(figsize=(10,10))
    for key in bsm_labels:

        true_label = np.concatenate(( np.ones(results['%s_scaled'%key].shape[0]), np.zeros(results['predicted_QCD'].shape[0]) ))
        pred_loss = np.concatenate(( results['mse_loss_%s'%key], results['mse_loss_QCD'] ))
        fpr_loss, tpr_loss, threshold_loss = roc_curve(true_label, pred_loss)

        auc_loss = auc(fpr_loss, tpr_loss)
        plt.plot(fpr_loss, tpr_loss, label='%s (AUC = %0.2f)' %(key,auc_loss))
	
    plt.legend(fontsize='x-small')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
    plt.title('ROC curve '+model_type)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(outdir+'/roc_curve_mse_loss_'+model_type+'.pdf')
    # plt.show()

    if(model_type=='VAE'):
        plt.figure(figsize=(10,10))
        for key in bsm_labels:

            true_label = np.concatenate(( np.ones(results['%s_scaled'%key].shape[0]), np.zeros(results['predicted_QCD'].shape[0]) ))
            pred_loss = np.concatenate(( results['total_loss_%s'%key], results['total_loss_QCD'] ))
            fpr_loss, tpr_loss, threshold_loss = roc_curve(true_label, pred_loss)

            auc_loss = auc(fpr_loss, tpr_loss)
            plt.plot(fpr_loss, tpr_loss, label='%s (AUC = %0.2f)' %(key,auc_loss))
	    
        plt.legend(fontsize='x-small')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
        plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
        plt.title('Total Loss ROC curve '+model_type)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(outdir+'/roc_curve_total_loss_'+model_type+'.pdf')
        # plt.show()

        plt.figure(figsize=(10,10))
        for key in bsm_labels:

            true_label = np.concatenate(( np.ones(results['%s_scaled'%key].shape[0]), np.zeros(results['predicted_QCD'].shape[0]) ))
            pred_loss = np.concatenate(( results['radius_%s'%key], results['radius_QCD'] ))
            fpr_loss, tpr_loss, threshold_loss = roc_curve(true_label, pred_loss)

            auc_loss = auc(fpr_loss, tpr_loss)
            plt.plot(fpr_loss, tpr_loss, label='%s (AUC = %0.2f)' %(key,auc_loss))
        plt.legend(fontsize='x-small')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
        plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
        plt.title('Radius ROC curve '+model_type)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(outdir+'/roc_curve_radius_'+model_type+'.pdf')
        # plt.show()

        plt.figure(figsize=(10,10))
        for key in bsm_labels:

            true_label = np.concatenate(( np.ones(results['%s_scaled'%key].shape[0]), np.zeros(results['predicted_QCD'].shape[0]) ))
            pred_loss = np.concatenate(( results['kl_loss_%s'%key], results['kl_loss_QCD'] ))
            fpr_loss, tpr_loss, threshold_loss = roc_curve(true_label, pred_loss)

            auc_loss = auc(fpr_loss, tpr_loss)
            plt.plot(fpr_loss, tpr_loss, label='%s (AUC = %0.2f)' %(key,auc_loss))
        plt.legend(fontsize='x-small')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
        plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
        plt.title('Radius ROC curve '+model_type)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(outdir+'/roc_curve_kl_loss_'+model_type+'.pdf')
        # plt.show()
	
    return 

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train aand test model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file with IO and training parameters setup')
    parser.add_argument('--run', type=str, default='both', help='what to run: train, eval, or all')
    parser.add_argument('--load_pickle', action="store_true", default=False, help='Load Directly from Pickle file')
    parser.add_argument('--model_quantize', action="store_true", default=False, help=' Use QKeras to Quantize Model ')
    parser.add_argument('--load_results', action="store_true", default=False, help='Load results')
    args=parser.parse_args()

    print("Loading configuration from", args.config)
    config = open(args.config, 'r')
    yamlConfig = yaml.load(config,Loader=yaml.FullLoader)

    if os.path.isdir(yamlConfig['OutDir']):
        input("Warning: output directory exists. Press Enter to continue...")
    else:
        os.mkdir(yamlConfig['OutDir'])

    if args.run=='train' or args.run=='all':
                 
     train(yamlConfig['InputQCD'],yamlConfig['InputBSM'],yamlConfig['DataFile'],
           yamlConfig['OutDir'],yamlConfig['Events'],yamlConfig['ModelType'],
	   yamlConfig['LatentDim'],yamlConfig['BatchSize'],yamlConfig['Epochs'],
	   args.load_pickle,args.model_quantize)
	   
    if args.run=='eval' or args.run=='all':
         	        
     evaluate(yamlConfig['InputQCD'],yamlConfig['InputBSM'],yamlConfig['DataFile'],
              yamlConfig['OutDir'],yamlConfig['Events'],yamlConfig['ModelType'],
	      yamlConfig['LatentDim'],yamlConfig['BatchSize'],yamlConfig['Epochs'],
	      args.load_pickle,args.model_quantize,args.load_results)


















    


    

    
