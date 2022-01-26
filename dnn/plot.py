import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import tensorflow as tf
import joblib
from losses import mse_split_loss, radius, kl_loss
from functions import make_mse_loss
from sklearn.metrics import roc_curve, auc

def read_data(results_file, model, beta=None):
    data = h5py.File(results_file, 'r')

    total_loss = []
    kl_data = []
    r_data = []
    mse_loss=[]
    
    X_test_scaled = data['QCD_target'][:]
    qcd_prediction = data['predicted_QCD'][:]
    #compute loss
    mse_loss.append(make_mse_loss(X_test_scaled, qcd_prediction.astype(np.float32)))
    if model=='VAE':
        qcd_mean = data['encoded_mean_QCD'][:]
        qcd_logvar = data['encoded_logvar_QCD'][:]
        qcd_z = data['encoded_z_QCD'][:]
        kl_data.append(kl_loss(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32), beta=beta))
        r_data.append(radius(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32)))
        
    #BSM
    for bsm in bsm_labels:
        print('read_data',bsm)
        bsm_target=data[bsm+'_scaled'][:]
        bsm_prediction=data['predicted_'+ bsm][:]
        mse_loss.append(make_mse_loss(bsm_target, bsm_prediction.astype(np.float32)))
        if model=='VAE':
            bsm_mean=data['encoded_mean_'+bsm][:]
            bsm_logvar=data['encoded_logvar_'+bsm][:]
            bsm_z=data['encoded_z_'+bsm][:]
            kl_data.append(kl_loss(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32), beta=beta))
            r_data.append(radius(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32)))
    if model=='VAE':
        total_loss=[]
        for mse, kl in zip(mse_loss, kl_data):
            total_loss.append(np.add(mse, kl))
    else:
        total_loss = mse_loss.copy()
    
    data.close()
    if model == 'VAE': del X_test_scaled, qcd_prediction, qcd_mean, qcd_logvar, qcd_z, bsm_target, bsm_prediction,\
                            bsm_mean, bsm_logvar, bsm_z
    else: del X_test_scaled, qcd_prediction, bsm_target, bsm_prediction

    return total_loss, mse_loss, kl_data, r_data

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

def return_label(anomaly):
    if anomaly == 'Leptoquark':
        marker = 'o'; sample_label=r'LQ $\rightarrow$ b$\tau$'
    elif anomaly == 'A to 4 leptons': 
        marker='X'; sample_label=r'A $\rightarrow$ 4L'
    elif anomaly == 'hToTauTau':
        marker = 'd'; sample_label=r'$h_{0} \rightarrow \tau\tau$'
    else:
        marker = 'v'; sample_label=r'$h_{\pm} \rightarrow \tau\nu$'
    return marker, sample_label

def plot_metric_easy(model, beta, save_name, anomaly='BSM'):

    # load AE results
    baseline_total_loss, _, _, _ = read_data(f'../../nobackup/DNN_AE_test_results.h5', model, beta)
        
    baseline_mse = []
    for i,bsm in enumerate(bsm_labels):
      #baseline_mse[bsm] = []
      print(i+1,bsm)
      get_metric(baseline_total_loss[0], baseline_total_loss[i+1])
    
      
    
def plot_metric(model1,model2, beta, pruned, save_name, anomaly='Leptoquark'):

    _, baseline_data_mse, baseline_data_kl, baseline_data_radius = read_data(f'/eos/user/e/epuljak/Documents/{model1}_result_pruned.h5', model1, beta)
    #
    #correct radius
    baseline_radius=list()
    for rad in baseline_data_radius:
        r = np.nan_to_num(rad);r[r==-np.inf] = 0;r[r==np.inf] = 0;r[r>=1E308] = 0;baseline_radius.append(r)

    baseline_lq_mse = get_metric(baseline_data_mse[0], baseline_data_mse[1])
    baseline_ato4l_mse = get_metric(baseline_data_mse[0], baseline_data_mse[2])
    baseline_hToTauTau_mse = get_metric(baseline_data_mse[0], baseline_data_mse[4])
    baseline_hChToTauNu_mse = get_metric(baseline_data_mse[0], baseline_data_mse[3])

    baseline_lq_kl = get_metric(baseline_data_kl[0], baseline_data_kl[1])
    baseline_ato4l_kl = get_metric(baseline_data_kl[0], baseline_data_kl[2])
    baseline_hToTauTau_kl = get_metric(baseline_data_kl[0], baseline_data_kl[4])
    baseline_hChToTauNu_kl = get_metric(baseline_data_kl[0], baseline_data_kl[3])
    #
    baseline_lq_radius = get_metric(baseline_radius[0], baseline_radius[1])
    baseline_ato4l_radius = get_metric(baseline_radius[0], baseline_radius[2])
    baseline_hToTauTau_radius = get_metric(baseline_radius[0], baseline_radius[4])
    baseline_hChToTauNu_radius = get_metric(baseline_radius[0], baseline_radius[3])

    fig = plt.figure(figsize=(15,10))

    labels = ['MeanSquaredError VAE', r'$D_{KL}$', r'$R_Z$', 'MeanSquaredError AE']
    #linestyles = ['solid', 'dashed', 'dotted']
    
    marker, sample_label = return_label(anomaly)
    
    if sample_label == r'LQ $\rightarrow$ b$\tau$':
        figures_of_merit = [baseline_lq_mse, baseline_lq_kl, baseline_lq_radius]
    elif sample_label == r'A $\rightarrow$ 4L':
        figures_of_merit = [baseline_ato4l_mse, baseline_ato4l_kl, baseline_ato4l_radius]
    elif sample_label == r'$h_{0} \rightarrow \tau\tau$':
        figures_of_merit = [baseline_hToTauTau_mse, baseline_hToTauTau_kl, baseline_hToTauTau_radius]
    elif sample_label == r'$h_{\pm} \rightarrow \tau\nu$':
        figures_of_merit = [baseline_hChToTauNu_mse, baseline_hChToTauNu_kl, baseline_hChToTauNu_radius]
    
    for i, base in enumerate(figures_of_merit):
        plt.plot(base[0], base[1], "-", label='%s: (auc = %.1f)'%(str(labels[i]) ,base[2]*100.), linewidth=1.5, color=colors[i+5])

    # load AE model
    baseline_total_loss, _, _, _ = read_data(f'AE_results/{model2}_result_pruned.h5', model2, beta)
    #

    baseline_lq_mse = get_metric(baseline_total_loss[0], baseline_total_loss[1])
    baseline_ato4l_mse = get_metric(baseline_total_loss[0], baseline_total_loss[2])
    baseline_hToTauTau_mse = get_metric(baseline_total_loss[0], baseline_total_loss[4])
    baseline_hChToTauNu_mse = get_metric(baseline_total_loss[0], baseline_total_loss[3])

    if sample_label == r'LQ $\rightarrow$ b$\tau$':
        figure_of_merit = baseline_lq_mse
    elif sample_label == r'A $\rightarrow$ 4L':
        figure_of_merit = baseline_ato4l_mse
    elif sample_label == r'$h_{0} \rightarrow \tau\tau$':
        figure_of_merit = baseline_hToTauTau_mse
    elif sample_label == r'$h_{\pm} \rightarrow \tau\nu$':
        figure_of_merit = baseline_hChToTauNu_mse
    plt.plot(figure_of_merit[0], figure_of_merit[1], "-", label='%s: (auc = %.1f)'% (str(labels[-1]), figure_of_merit[2]*100.), linewidth=1.5, color='C8')
    
    
    plt.yscale('log', nonpositive='clip')
    plt.xscale('log', nonpositive='clip')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
    plt.title('ROC '+ sample_label)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True)
    #plt.show()
    plt.savefig('VAE_rocs'+ str(save_name)+ '.pdf', dpi = fig.dpi, bbox_inches='tight')
        
if __name__ == "__main__":
    bsm_labels = ['Leptoquark','A to 4 leptons', 'hChToTauNu', 'hToTauTau']
    bsm_labels = ['VectorZPrimeToQQ__M50']
    colors = ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    for bsm in bsm_labels:
        #plot_metric('VAE', 'AE', 0.8, '5to15', '_result_'+bsm+'_pruned', bsm)
        plot_metric_easy('AE', 0.8, 'test', bsm_labels)
