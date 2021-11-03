import argparse
import h5py
import numpy as np
import tensorflow as tf
import pickle
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, QActivation
from models import model_set_weights

import setGPU

def evaluate(input_h5, input_json, input_file, input_history,
    output_result, quant_size):
    # magic trick to make sure that Lambda function works
    tf.compat.v1.disable_eager_execution()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # load dataset and pt scaler
    with open(input_file, 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

    # load history
    with open(input_history, 'rb') as f:
        history = pickle.load(f)

    # load trained model
    with open(input_json, 'r') as jsonfile:
        config = jsonfile.read()
    model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    if quant_size:
        model = model_set_weights(model, input_h5.strip('.h5'), quant_size)
    else:
        model.load_weights(input_h5)

    # load encoder
    encoder = model.layers[1]

    # get prediction
    predicted_QCD = model.predict(x_test)
    encoded_QCD = encoder.predict(x_test)

    # test model on BSM data
    result_bsm = []
    for i, bsm_data_name in enumerate(['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']):
        bsm_data = all_bsm_data[i]
        predicted_bsm_data = model.predict(bsm_data)
        encoded_bsm = encoder.predict(bsm_data)
        bsm_data = np.squeeze(bsm_data, axis=-1)
        bsm_data_target = np.copy(bsm_data)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm_data[:,:,0],0))
        bsm_data_target = bsm_data_target.reshape(bsm_data_target.shape[0],bsm_data_target.shape[1],bsm_data_target.shape[2],1)
        result_bsm.append([bsm_data_name, predicted_bsm_data, bsm_data_target, encoded_bsm[0], encoded_bsm[1], encoded_bsm[2]])

    #Save results
    with h5py.File(output_result, 'w') as h5f:
        h5f.create_dataset('loss', data=history['loss'])
        h5f.create_dataset('val_loss', data=history['val_loss'])
        h5f.create_dataset('QCD', data=y_test)
        h5f.create_dataset('predicted_QCD', data=predicted_QCD)
        h5f.create_dataset('encoded_mean_QCD', data=encoded_QCD[0])
        h5f.create_dataset('encoded_logvar_QCD', data=encoded_QCD[1])
        h5f.create_dataset('encoded_z_QCD', data=encoded_QCD[2])
        for bsm in result_bsm:
            h5f.create_dataset(f'{bsm[0]}_scaled', data=bsm[2])
            h5f.create_dataset(f'predicted_{bsm[0]}', data=bsm[1])
            h5f.create_dataset(f'encoded_mean_{bsm[0]}', data=bsm[3])
            h5f.create_dataset(f'encoded_logvar_{bsm[0]}', data=bsm[4])
            h5f.create_dataset(f'encoded_z_{bsm[0]}', data=bsm[5])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-h5', type=str, help='Where is the model')
    parser.add_argument('--input-json', type=str, help='Where is the model')
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--input-history', type=str, help='input history file', required=True)
    parser.add_argument('--quant-size', default=None, type=int, help='Train quantized model with QKeras')
    parser.add_argument('--output-result', type=str, help='Output file with results', required=True)
    args = parser.parse_args()
    evaluate(**vars(args))
