import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
from models import (
    conv_vae,
    conv_ae
    )
import tensorflow_model_optimization as tfmot
import pickle
import setGPU

def train(model_type, quant_size, pruning, latent_dim, output_model_h5,
    output_model_json, output_history, batch_size,
    n_epochs, beta):
    # magic trick to make sure that Lambda function works
    tf.compat.v1.disable_eager_execution()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    with open('output/data_-1.pickle', 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

    if model_type=='conv_vae':
        model = conv_vae(x_train.shape, latent_dim, beta, quant_size, pruning)
    elif model_type=='conv_ae':
        model = conv_ae(x_train.shape, latent_dim, quant_size, pruning)

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
        ]
    if pruning=='pruned':
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    # train
    history = model.fit(x=x_train, y=y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks)

    # save model
    model_json = model.to_json()
    with open(output_model_json, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(output_model_h5)
    print('The latent size is ', latent_dim)

    # save training history
    with open(output_history, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='conv_vae',
        choices=['conv_vae', 'conv_ae'], help='Which model to use')
    parser.add_argument('--quant-size', default=0, type=int, help='Train quantized model with QKeras')
    parser.add_argument('--pruning', type=str, help='Train with pruning')
    parser.add_argument('--latent-dim', type=int, required=True, help='Latent space dimension')
    parser.add_argument('--output-model-h5', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-model-json', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-history', type=str, help='Output file with the model training history', required=True)
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--beta', type=float, default=1.0, help='Fraction of KL loss')
    args = parser.parse_args()
    train(**vars(args))
