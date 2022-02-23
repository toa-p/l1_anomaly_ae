import argparse
from keras.models import load_model
import tensorflow as tf
import pickle
import h5py

from autoencoder_classes import AE

def evaluate(input_h5, input_json, input_file, input_history, output_result):

    with open(input_file, 'rb') as f:
       X_train_flatten, X_train_scaled, X_test_flatten, X_test_scaled, bsm_data, bsm_target, pt_scaler = pickle.load(f)
      
    # load trained model
    with open(input_json, 'r') as jsonfile:
        config = jsonfile.read()
    model = tf.keras.models.model_from_json(config)    
    model.load_weights(input_h5)
    model.summary()

    # load history
    with open(input_history, 'rb') as f:
        print("*** Loaded the data")
        history = pickle.load(f)
	    
    ae = AE(model)
    qcd_prediction = ae.autoencoder(X_test_flatten)

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

    bsm_results = []
    for i, label in enumerate(bsm_labels):
        bsm_prediction = ae.autoencoder(bsm_data[i])    
        bsm_results.append([label, bsm_target[i], bsm_prediction])

    h5f = h5py.File(output_result, 'w')
    h5f.create_dataset('loss', data=history['loss'])
    h5f.create_dataset('val_loss', data=history['val_loss'])
    h5f.create_dataset('QCD_input', data=X_test_flatten)
    h5f.create_dataset('QCD_target', data=X_test_scaled)
    h5f.create_dataset('predicted_QCD', data = qcd_prediction)
    for i, bsm in enumerate(bsm_results):
       h5f.create_dataset('%s_scaled' %bsm[0], data=bsm[1])
       h5f.create_dataset('%s_input' %bsm[0], data=bsm_data[i])
       h5f.create_dataset('predicted_%s' %bsm[0], data=bsm[2])
    print("*** OutputFile Created")
    h5f.close()
    		  
    #plt.figure()
    #plt.plot(history.history['loss'][:], label='Training loss')
    #plt.plot(history.history['val_loss'][:], label='Validation loss')
    #plt.title('Training and validation loss - MSE')
    ##plt.yscale('log', nonposy='clip')
    #plt.legend(loc='best')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.savefig(input_history.replace('.h5','.pdf'))
    #plt.show()
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-h5', type=str, help='Where is the model')
    parser.add_argument('--input-json', type=str, help='Where is the model')
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--input-history', type=str, help='input history file', required=True)
    parser.add_argument('--output-result', type=str, help='Output file with results', required=True)
    args = parser.parse_args()
    evaluate(**vars(args))
