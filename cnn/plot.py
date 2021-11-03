### L1 Anomaly conv autoenc - plot results - v2.4 - added version control
#used for v3.4 - v3.5
import os
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import argparse
from sklearn.manifold import TSNE

def make_feature_plot(coord, feature_index, predicted_qcd, X_test, particle_name, loss_type, output_dir):
    input_featurenames = ['pT', 'eta', 'phi'] if coord=='cyl' else ['px', 'py', 'pz']

    true0 = X_test[:,:,feature_index].flatten()
    true = [i for i in true0 if i!=0]
    zeroes = [i for i,v in enumerate(true0) if v==0]
    print(len(true0), len(true))

    predicted0 = predicted_qcd[:,:,feature_index].flatten()
    predicted = np.delete(predicted0, zeroes)
    print(len(predicted0), len(predicted))
    if feature_index==2:
        predicted = np.cos(predicted)*math.pi

    # then plot the right quantity for the reduced array
    plt.hist(predicted, 100, density=True, histtype='step', fill=False, linewidth=1.5, label=f'{particle_name} Predicted')
    plt.hist(true, 100, density=True, histtype='step', fill=False, linewidth=1.5, label=f'{particle_name} True')

    plt.yscale('log', nonpositive='clip')
    plt.legend(fontsize=12, frameon=False)
    # plt.xlim(10,100)
    plt.xlabel(str(input_featurenames[feature_index]), fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    plt.tight_layout()
    if feature_index==2: plt.ylim(bottom=0.01)
    plt.savefig(os.path.join(output_dir,f'VAE_{loss_type}_{particle_name}_{input_featurenames[feature_index]}.pdf'))
    plt.show()
    plt.clf()

def mse_loss(inputs, outputs):
    mse = (inputs-outputs)*(inputs-outputs)
    mse = np.sum(mse, axis=-1)
    if len(outputs.shape)>=3:
        mse = np.sum(mse, axis=-1)
    if len(outputs.shape)==4:
        mse = np.sum(mse, axis=-1)
    return mse

def kl_loss(z_mean, z_log_var):
    kl = np.ones(z_mean.shape) + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = - 0.5 * np.mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)
    return kl

def threeD_loss(inputs, outputs):
    if len(outputs.shape)==4:
        #[batch_size x 100 x 3 x 1] -- remove last dim
        inputs = np.squeeze(inputs, axis=-1)
        outputs = np.squeeze(outputs, axis=-1)
    expand_inputs = np.expand_dims(inputs, 2) # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_outputs = np.expand_dims(outputs, 1) # add broadcasting dim [batch_size x 1 x 100 x 3]
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100]
    # where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = np.sum((expand_outputs-expand_inputs)*(expand_outputs-expand_inputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = np.ndarray.min(distances,1)
    min_dist_to_outputs = np.ndarray.min(distances,2)

    return (np.sum(min_dist_to_inputs, 1) + np.sum(min_dist_to_outputs, 1))/outputs.shape[1]/outputs.shape[2]


def mse_loss(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def kl_loss(z_mean, z_log_var):
    kl = 1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = - 0.5 * np.mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)
    return kl

def compute_loss(inputs, outputs, z_mean=None, z_log_var=None, beta=None):
    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    loss = mse_loss(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))

    reco_loss = np.copy(loss)
    kl = None
    if z_mean is not None:
        kl = kl_loss(z_mean, z_log_var)
        loss = reco_loss + kl if beta==0 else (1-beta)*reco_loss + beta*kl

    return loss, kl, reco_loss

# Latent space radius
def radius(mean, logvar):
    sigma = np.sqrt(np.exp(logvar))
    radius = mean*mean/sigma/sigma
    print('Radius shape:', radius.shape)
    return np.sum(radius, axis=-1)

def plot_vae(coord, model, loss_type, output_dir, input_dir, label):
    if model=='dae': loss_type = 'mse'
    #read in data
    h5f = h5py.File(os.path.join(input_dir, 'result.h5'), 'r')

    samples = ['QCD','VectorZPrimeToQQ__M50',
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

    labels = ['QCD multijet', 'VectorZPrimeToQQ__M50',
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


    colors = ['#238b45', '#4eb3d3', '#2b8cbe', '#08589e', '#ef6548', '#d7301f', '#990000', '#ce1256', '#525252']

    mse_kl_data = []
    mse_data = []
    kl_data = []
    r_data = []
    z_anomaly = []

    for sample in samples:
        inval = np.array(h5f[sample])
        outval = np.array(h5f['predicted_'+sample])
        meanval = np.array(h5f['encoded_mean_'+sample])
        logvarval = np.array(h5f['encoded_logvar_'+sample])
        mseklval, kl, mseval = compute_loss(inval, outval, meanval, logvarval, beta) \
            if model=='conv_vae' or model=='dense_vae' else compute_loss(inval, outval)
        mse_kl_data.append(mseklval)
        mse_data.append(mseval)
        if model=='conv_vae' or model=='dense_vae':
            kl_data.append(kl)
            r_data.append(radius(meanval, logvarval))

    loss = h5f['loss'][:]
    val_loss = h5f['val_loss'][:]

    #--------
    #Training history
    #--------
    print('Training history plot')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training History')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir+loss_type+'_training_hist.pdf')
    plt.show()

    plt.figure()
    print('Reco loss plot')
    for i, label in enumerate(labels):
        plt.hist(mse_data[i], bins=100, label=label, density=True,
             histtype='step', fill=False, linewidth=1.5)
    # #plt.semilogx()
    # if '3D' in loss_type:
    #     plt.xlim(0,20)
    # else:
    #     plt.xlim(0,500)
    plt.semilogy()
    plt.title(loss_type)
    plt.xlabel('Autoencoder Loss')
    plt.ylabel('Probability (a.u.)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir+'/'+label+'_loss_dist.pdf')
    plt.show()

    minScore = 999999.
    maxScore = 0.
    # for i in range(len(labels)):
    #     thisMin = np.min(radius_data[i])
    #     thisMax = np.max(radius_data[i])
    #     minScore = min(thisMin, minScore)
    #     maxScore = max(maxScore, thisMax)
    # ----------
    # ROC Curves
    # ----------
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(9,5))
    print('Reco vs $R^2$ ROC plot')
    target_qcd = np.zeros(mse_data[0].shape[0])
    for i, label in enumerate(labels):
        if i == 0: continue
        trueVal = np.concatenate((np.ones(mse_data[i].shape[0]), target_qcd))
        predVal_loss = np.concatenate((mse_data[i], mse_data[0]))

        fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

        auc_loss = auc(fpr_loss, tpr_loss)

        plt.plot(fpr_loss, tpr_loss, '-', label='%s MSE ROC (auc = %.1f%%)'%(label,auc_loss*100.), linewidth=1.5)
        plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')

    plt.xlim(10**(-6),1)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1,0.815), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir+'/'+model+loss_type+'_ROC.pdf')
    plt.show()
    plt.clf()

    # # #----------
    # # Features
    # #----------
    # X_test = h5f['QCD'][:]
    # predicted_qcd = h5f['predicted_QCD'][:]

    # # we now plot all the features
    # for i in range(3):
    #     make_feature_plot(coord, i, predicted_qcd[:,0:10], X_test[:,0:10], 'Jets', loss_type, output_dir)
    #     make_feature_plot(coord, i, predicted_qcd[:,10:14], X_test[:,10:14], 'Muons', loss_type, output_dir)
    #     make_feature_plot(coord, i, predicted_qcd[:,14:18], X_test[:,14:18], 'Electrons', loss_type, output_dir)

    # #--------------
    # # Latent space
    # #--------------
    # idx_max = 5000
    # z_dset = h5f['encoded_mean_QCD'][:idx_max]
    # for i in range(len(labels)-1):
    #     z_dset = np.concatenate((z_dset, z_anomaly[i+1][:idx_max]))
    # z_embedded = TSNE(n_components=2).fit_transform(z_dset)

    # f = plt.figure(figsize=(8,8))
    # for i, n in enumerate(labels):
    #     if i != 0 : continue
    #     aux_z = z_embedded[i*idx_max: (i+1)*idx_max]
    #     plt.plot(aux_z[:,0], aux_z[:,1],
    #             'o', mfc='none', label=n)
    # plt.xlabel('Embedded 0')
    # plt.ylabel('Embedded 1')
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(output_dir, f'latent_space_{loss_type}.pdf'))
    # plt.clf()

    # for i, n in enumerate(labels):
    #     if 'QCD' in n: continue
    #     qcd_z = z_embedded[0: idx_max]
    #     aux_z = z_embedded[i*idx_max: (i+1)*idx_max]
    #     plt.plot(qcd_z[:,0], qcd_z[:,1], 'o', mfc='none', label='QCD')
    #     plt.plot(aux_z[:,0], aux_z[:,1], 'o', mfc='none', label=n)
    #     plt.xlabel('Embedded 0')
    #     plt.ylabel('Embedded 1')
    #     plt.legend(loc='best')
    #     plt.savefig(os.path.join(output_dir, f'latent_space_{loss_type}_{n}.pdf'))
    #     plt.clf()

    h5f.close()


def plot_ae(coord, model, loss_type, output_dir, input_dir, label):
    #read in data
    h5f = h5py.File(os.path.join(input_dir, 'result.h5'), 'r')

    samples = ['VectorZPrimeToQQ__M50',
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

    labels = ['VectorZPrimeToQQ__M50',
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

    colors = ['#238b45', '#4eb3d3', '#2b8cbe', '#08589e', '#ef6548', '#d7301f', '#990000', '#ce1256', '#525252']

    mse_data = []
    z_anomaly = []
    for sample in samples:
        inval = np.array(h5f[sample])
        outval = np.array(h5f['predicted_'+sample])
        mseklval, _, _ = compute_loss(inval, outval)
        mse_data.append(mseklval)

    loss = h5f['loss'][:]
    val_loss = h5f['val_loss'][:]

    #--------
    #Training history
    #--------
    print('Training history plot')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training History')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir+f'{model}_training_hist.pdf')
    plt.show()

    bin_size = 100

    minScore = 999999.
    maxScore = 0.
    for i in range(len(labels)):
        thisMin = np.min(mse_data[i])
        thisMax = np.max(mse_data[i])
        minScore = min(thisMin, minScore)
        maxScore = max(maxScore, thisMax)

    plt.figure()
    print('Reco loss plot')
    for i, label in enumerate(labels):
        plt.hist(mse_data[i], bins=bin_size, label=label, density=True, range=(minScore, maxScore),
             histtype='step', fill=False, linewidth=1.5)
    #plt.semilogx()
    plt.semilogy()
    # if '3D' in loss_type:
    #     plt.xlim(0,100)
    # else:
    #     plt.xlim(0,500)
    plt.title(loss_type)
    plt.xlabel('Autoencoder Loss')
    plt.ylabel('Probability (a.u.)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir+'/'+label+'_loss_dist.pdf')
    plt.show()

    # ----------
    # ROC Curves
    # ----------
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(9,5))
    print('Reco vs $R^2$ ROC plot')
    for i, label in enumerate(labels):
        if i == 0: continue
        trueVal = np.concatenate((np.ones(mse_data[i].shape[0]), np.zeros(mse_data[0].shape[0])))
        predVal_loss = np.concatenate((mse_data[i], mse_data[0]))

        fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

        auc_loss = auc(fpr_loss, tpr_loss)

        plt.plot(fpr_loss, tpr_loss, '-', label='%s ROC (auc = %.1f%%)'%(label,auc_loss*100.), linewidth=1.5)
        plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')

    plt.xlim(10**(-6),1)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1,0.815), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir+'/'+label+'_ROC.pdf')
    plt.show()
    plt.clf()

    # #----------
    # # Features
    # #----------
    # X_test = h5f['QCD'][:]
    # predicted_qcd = h5f['predicted_QCD'][:]

    # # we now plot all the features
    # for i in range(3):
    #     make_feature_plot(coord, i, predicted_qcd[:,0:10,:], X_test[:,0:10,:], 'Jets', loss_type, output_dir)
    #     make_feature_plot(coord, i, predicted_qcd[:,10:14,:], X_test[:,10:14,:], 'Muons', loss_type, output_dir)
    #     make_feature_plot(coord, i, predicted_qcd[:,14:18,:], X_test[:,14:18,:], 'Electrons', loss_type, output_dir)


    # for i, label in enumerate(labels):
    #     # we now plot all the features
    #     for feat in range(3):
    #         make_feature_plot(coord, feat, bsm_in[i][:,0:10,:], bsm_out[i][:,0:10,:], 'Jets', loss_type, output_dir)
    #         make_feature_plot(coord, feat, bsm_in[i][:,10:14,:], bsm_out[i][:,10:14,:], 'Muons', loss_type, output_dir)
    #         make_feature_plot(coord, feat, bsm_in[i][:,14:18,:], bsm_out[i][:,14:18,:], 'Electrons', loss_type, output_dir)

    # #--------------
    # # Latent space
    # #--------------
    # idx_max = 5000
    # z_dset = h5f['encoded_mean_QCD'][:idx_max]
    # for i in range(len(labels)-1):
    #     z_dset = np.concatenate((z_dset, z_anomaly[i+1][:idx_max]))
    # z_embedded = TSNE(n_components=2).fit_transform(z_dset)

    # f = plt.figure(figsize=(8,8))
    # for i, n in enumerate(labels):
    #     if i != 0 : continue
    #     aux_z = z_embedded[i*idx_max: (i+1)*idx_max]
    #     plt.plot(aux_z[:,0], aux_z[:,1],
    #             'o', mfc='none', label=n)
    # plt.xlabel('Embedded 0')
    # plt.ylabel('Embedded 1')
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(output_dir, f'latent_space_{loss_type}.pdf'))
    # plt.clf()

    # for i, n in enumerate(labels):
    #     if 'QCD' in n: continue
    #     qcd_z = z_embedded[0: idx_max]
    #     aux_z = z_embedded[i*idx_max: (i+1)*idx_max]
    #     plt.plot(qcd_z[:,0], qcd_z[:,1], 'o', mfc='none', label='QCD')
    #     plt.plot(aux_z[:,0], aux_z[:,1], 'o', mfc='none', label=n)
    #     plt.xlabel('Embedded 0')
    #     plt.ylabel('Embedded 1')
    #     plt.legend(loc='best')
    #     plt.savefig(os.path.join(output_dir, f'latent_space_{loss_type}_{n}.pdf'))
    #     plt.clf()

    h5f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coord', type=str, default='cyl', help='Use cylindrical (cyl) or cartesian (cart) coordinates')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'cae', 'qcae', 'dae'], required=True, help='Which model was used')
    parser.add_argument('--loss-type', type=str, default='mse_kl',
        choices=['mse', 'mse_kl', '3D_kl', 'gauss_kl', 'split_mse', 'split_3D', 'split_mse_kl', 'split_3D_kl'],
        required=True, help='Which loss function to use')
    parser.add_argument('--output-dir', type=str, help='output directory', required=True)
    parser.add_argument('--input-dir', type=str, help='input directory', required=True)
    parser.add_argument('--label', type=str, default='', help='model label')
    args = parser.parse_args()
    if 'vae' in args.model:
        plot_vae(**vars(args))
    else:
        plot_ae(**vars(args))

