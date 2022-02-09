import argparse
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

# import setGPU
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K

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
from functions import make_mse_loss_numpy
from sklearn.metrics import roc_curve, auc


from data_preprocessing import prepare_data
from model import build_AE, build_VAE





def return_total_loss(loss, bsm_t, bsm_pred):
    total_loss = loss(bsm_t, bsm_pred.astype(np.float32))
    return total_loss



def run_all(input_qcd='',input_bsm='',events=10000,load_pickle=False,input_pickle='',output_pfile='',model_type='AE',latent_dim=3,output_model_h5='',output_model_json='',output_history='',batch_size= 1024,n_epochs = 20,output_result='',tag=''):

    if(load_pickle):
        if(input_pickle==''):
            print('Please provide input pickle files')
            return
        with open(input_pickle, 'rb') as f:
            X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler = pickle.load(f)
            bsm_labels=['VectorZPrimeToQQ__M50',
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
    else:
        if(input_qcd==''or input_bsm==''):
            print('Please provide input H5 files')
            return
        X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels = prepare_data(input_qcd, input_bsm, events, '',True)

    if(output_pfile!=''):
        with open(output_pfile, 'wb') as f:
            pickle.dump([X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler, bsm_labels], f)
        print("Saved Pickle data to disk")
    
    if(model_type=='AE'):
        autoencoder = build_AE(X_train_flatten.shape[-1],latent_dim)
        model = AE(autoencoder)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001))

        callbacks=[]
        callbacks.append(ReduceLROnPlateau(monitor='val_loss',  factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6))
        callbacks.append(TerminateOnNaN())
        callbacks.append(NeptuneMonitor())
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=10, restore_best_weights=True))
    
    elif(model_type=='VAE'):
        encoder, decoder = build_VAE(X_train_flatten.shape[-1],latent_dim)
        model = VAE(encoder, decoder)
        model.compile(optimizer=keras.optimizers.Adam())

        callbacks=[]
        callbacks.append(ReduceLROnPlateau(monitor='val_loss',  factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6))
        callbacks.append(TerminateOnNaN())
        callbacks.append(NeptuneMonitor())
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=10, restore_best_weights=True))

    print("Training the model")

    history = model.fit(X_train_flatten, X_train_scaled,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=callbacks)

    if(output_model_h5!=''):
        if(model_type=='VAE'):
            model.save(os.path.join(os.getcwd(),output_model_h5.split('.')[0]))
        else:
            model_json = autoencoder.to_json()
            with open(output_model_json, 'w') as json_file:
                json_file.write(model_json)
            autoencoder.save_weights(output_model_h5)
            print("Saved model to disk")


    if(output_history!=''):
        with open(output_history, 'wb') as f:
            pickle.dump(history.history, f)
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
    plt.savefig('loss.pdf')
    # plt.show()

    # Evaluate the model
    print("Evaluating the model")
    loss = make_mse_loss_numpy
    if(model_type=='AE'): 
        qcd_prediction = model.autoencoder(X_test_scaled).numpy()
    elif(model_type=='VAE'):
        qcd_mean, qcd_logvar, qcd_z = model.encoder(X_test_scaled)
        qcd_prediction = model.decoder(qcd_z).numpy()


    results={}
    for i, label in enumerate(bsm_labels):
        results[label] = {}
        if(model_type=='AE'): 
            bsm_pred = model.autoencoder(bsm_data[i]).numpy()
        elif(model_type=='VAE'): 
            mean_pred, logvar_pred, z_pred = model.encoder(bsm_data[i])
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
        results['QCD']['mean_prediction'] = qcd_mean.numpy()
        results['QCD']['logvar_prediction'] = qcd_logvar.numpy()
        results['QCD']['z_prediction'] = qcd_z.numpy()
        results['QCD']['kl_loss'] = np.nan_to_num(kl_loss(qcd_mean.numpy(),qcd_logvar.numpy()))
        results['QCD']['total_loss']=np.nan_to_num(kl_loss(qcd_mean.numpy(),qcd_logvar.numpy())+qcd_loss)
        results['QCD']['radius']=np.nan_to_num(radius(qcd_mean.numpy(),qcd_logvar.numpy()))

    min_loss,max_loss=1e5,0
    if(model_type=='VAE'):
        min_tloss,max_tloss=1e5,0
        min_r,max_r=1e5,0
    for key in results.keys():
        if(key=='QCD'): continue
        if(np.min(results[key]['loss'])<min_loss): min_loss = np.min(results[key]['loss'])
        if(np.max(results[key]['loss'])>max_loss): max_loss = np.max(results[key]['loss'])
        if(model_type=='VAE'):
            if(np.min(results[key]['total_loss'])<min_tloss): min_tloss = np.min(results[key]['total_loss'])
            if(np.max(results[key]['total_loss'])>max_tloss): max_tloss = np.max(results[key]['total_loss'])
            if(max_tloss>np.mean(results[key]['total_loss'])+10*np.std(results[key]['total_loss'])): max_tloss = np.mean(results[key]['total_loss'])+10*np.std(results[key]['total_loss'])
            if(np.min(results[key]['radius'])<min_r): min_r = np.min(results[key]['radius'])
            if(np.max(results[key]['radius'])>max_r): max_r = np.max(results[key]['radius'])
            if(max_r>np.mean(results[key]['radius'])+10*np.std(results[key]['radius'])): max_r = np.mean(results[key]['radius'])+10*np.std(results[key]['radius'])


    if(output_result!=''):
        h5f = h5py.File(output_result, 'w')
        h5f.create_dataset('loss', data=hist['loss'].to_numpy())
        h5f.create_dataset('val_loss', data=hist['val_loss'].to_numpy())
        h5f.create_dataset('QCD_input', data=X_test_flatten)
        h5f.create_dataset('QCD_target', data=X_test_scaled)
        h5f.create_dataset('predicted_QCD', data = qcd_prediction)
        if(model_type=='VAE'):
            h5f.create_dataset('encoded_mean_QCD', data=qcd_mean.numpy())
            h5f.create_dataset('encoded_logvar_QCD', data=qcd_logvar.numpy())
            h5f.create_dataset('encoded_z_QCD', data=qcd_z.numpy())
        for i, key in enumerate(results):
            if(key=='QCD'): continue
            h5f.create_dataset('%s_scaled' %key, data=results[key]['target'])
            h5f.create_dataset('%s_input' %key, data=bsm_data[i])
            h5f.create_dataset('predicted_%s' %key, data=results[key]['prediction'])
            if(model_type=='VAE'):
                h5f.create_dataset('encoded_mean_%s' %key, data=results[key]['mean_prediction'])
                h5f.create_dataset('encoded_logvar_%s' %key, data=results[key]['logvar_prediction'])
                h5f.create_dataset('encoded_z_%s' %key, data=results[key]['z_prediction'])
        print("*** OutputFile Created")
        h5f.close() 


    # Plot the results
    print("Plotting the results")
    bins_=np.linspace(min_loss,max_loss,100)
    plt.figure(figsize=(10,10))
    for key in results.keys():
        if(key=='QCD'): plt.hist(results[key]['loss'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
        else: plt.hist(results[key]['loss'],label=key,histtype='step',bins=bins_,density=True)
    plt.legend(fontsize='x-small')
    plt.yscale('log')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.title('Loss distribution')
    plt.savefig('loss_hist_'+model_type+'_'+tag+'.pdf')
    # plt.show()

    if(model_type=='VAE'):

        bins_=np.linspace(min_tloss,10000,100)
        plt.figure(figsize=(10,10))
        for key in results.keys():
            if(key=='QCD'): plt.hist(results[key]['total_loss'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
            else: plt.hist(results[key]['total_loss'],label=key,histtype='step',bins=bins_,density=True)
        plt.legend(fontsize='x-small')
        plt.yscale('log')
        plt.xlabel('Total Loss')
        plt.ylabel('Density')
        plt.title('Total Loss distribution')
        plt.savefig('total_loss_hist_'+model_type+'_'+tag+'.pdf')

        bins_=np.linspace(min_r,1500,100)
        plt.figure(figsize=(10,10))
        for key in results.keys():
            if(key=='QCD'): plt.hist(results[key]['radius'],label=key,histtype='step',bins=bins_,color='black',linewidth=2,density=True)
            else: plt.hist(results[key]['radius'],label=key,histtype='step',bins=bins_,density=True)
        plt.legend(fontsize='x-small')
        plt.yscale('log')
        plt.xlabel('Radius')
        plt.ylabel('Density')
        plt.title('Radius distribution')
        plt.savefig('radius_hist_'+model_type+'_'+tag+'.pdf')
        
        for key in results.keys():
            plt.figure(figsize=(10,10))
            for i in range(latent_dim):
                plt.hist(results[key]['mean_prediction'][:,i],bins=100,label='mean '+str(i),histtype='step', density=True,range=[-5,5])
            plt.legend(fontsize='x-small')
            plt.xlabel('Loss')
            plt.ylabel('z')
            plt.title(key+' mean Z distribution')
            plt.savefig('mean_z_'+model_type+'_'+key+'_'+tag+'.pdf')
            # plt.show()

        for key in results.keys():
            plt.figure(figsize=(10,10))
            for i in range(latent_dim):
                plt.hist(results[key]['logvar_prediction'][:,i],bins=100,label='logvar '+str(i),histtype='step', density=True,range=[-20,20])
            plt.legend(fontsize='x-small')
            plt.xlabel('Loss')
            plt.ylabel('z')
            plt.title(key+' logvar Z distribution')
            plt.savefig('logvar_z_'+model_type+'_'+key+'_'+tag+'.pdf')
            # plt.show()

    




    plt.figure(figsize=(10,10))
    for key in results.keys():
        if key=='QCD': continue

        true_label = np.concatenate(( np.ones(results[key]['target'].shape[0]), np.zeros(results['QCD']['prediction'].shape[0]) ))
        pred_loss = np.concatenate(( results[key]['loss'], results['QCD']['loss'] ))
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
    plt.savefig('roc_curve_'+model_type+'_'+tag+'.pdf')
    # plt.show()

    if(model_type=='VAE'):
        plt.figure(figsize=(10,10))
        for key in results.keys():
            if key=='QCD': continue

            true_label = np.concatenate(( np.ones(results[key]['target'].shape[0]), np.zeros(results['QCD']['prediction'].shape[0]) ))
            pred_loss = np.concatenate(( results[key]['total_loss'], results['QCD']['total_loss'] ))
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
        plt.savefig('roc_curve_total_loss_'+model_type+'_'+tag+'.pdf')
        # plt.show()

        plt.figure(figsize=(10,10))
        for key in results.keys():
            if key=='QCD': continue

            true_label = np.concatenate(( np.ones(results[key]['target'].shape[0]), np.zeros(results['QCD']['prediction'].shape[0]) ))
            pred_loss = np.concatenate(( results[key]['radius'], results['QCD']['radius'] ))
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
        plt.savefig('roc_curve_radius_'+model_type+'_'+tag+'.pdf')
        # plt.show()


    return 

if __name__ == '__main__':
    #(input_qcd='',input_bsm='',events,load_pickle=False,input_pickle='',output_pfile='',model_type='AE',latent_dim,output_model_h5='',output_model_json='',output_history='',batch_size= 1024,n_epochs = 20,output_result='',tag='')
    parser=argparse.ArgumentParser(description='Train aand test model')
    parser.add_argument('--input_qcd', type=str, default='', help='Input QCD file')
    parser.add_argument('--input_bsm', type=str, default='', help='Input BSM file')
    parser.add_argument('--events', type=int, default=0, help='Number of events to process')
    parser.add_argument('--load_pickle', type=bool, default=False, help='Load Directly from Pickle file')
    parser.add_argument('--input_pickle', type=str, default='', help='Input Pickle file')
    parser.add_argument('--output_pfile', type=str, default='', help='Output Pickle file')
    parser.add_argument('--model_type', type=str, default='AE',choices=['AE','VAE'], help='Model type')
    parser.add_argument('--latent_dim', type=int, default=3, help='Latent dimension')
    parser.add_argument('--output_model_h5', type=str, default='', help='Output model h5 file')
    parser.add_argument('--output_model_json', type=str, default='', help='Output model json file')
    parser.add_argument('--output_history', type=str, default='', help='Output history file')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--output_result', type=str, default='', help='Output result file')
    parser.add_argument('--tag', type=str, default='', help='Tag for output files')
    args=parser.parse_args()
    run_all(args.input_qcd,args.input_bsm,args.events,args.load_pickle,args.input_pickle,args.output_pfile,args.model_type,args.latent_dim,args.output_model_h5,args.output_model_json,args.output_history,args.batch_size,args.n_epochs,args.output_result,args.tag)


















    


    

    