import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Lambda,
    Input,
    Dense,
    LeakyReLU,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Reshape,
    Activation,
    Concatenate,
    Cropping1D
    )
from qkeras import (
    QConv2D,
    QDense,
    QActivation,
    QInitializer
    )
import tensorflow_model_optimization as tfmot

from losses import (
    make_mse_kl,
    make_mse,
    make_kl
    )

# number of integer bits for each bit width
QUANT_INT = {
    0: 0,
    2: 1,
    4: 2,
    6: 2,
    8: 3,
    10: 3,
    12: 4,
    14: 4,
    16: 6
    }

def model_set_weights(model, load_model, quant_size):
   # load trained model
    with open(load_model+'.json', 'r') as jsonfile:
        config = jsonfile.read()
    bp_model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
        'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    bp_model.load_weights(load_model+'.h5')

    # set weights for encoder and skip input quantization
    if quant_size!=0:
        for i, _ in enumerate(model.layers[1].layers):
                if i < 2: continue
                model.layers[1].layers[i].set_weights(bp_model.layers[1].layers[i-1].get_weights())
    else:
        for i, _ in enumerate(model.layers[1].layers):
                model.layers[1].layers[i].set_weights(bp_model.layers[1].layers[i].get_weights())
    # set weights for decoder
    for i, _ in enumerate(model.layers[2].layers):
        model.layers[2].layers[i].set_weights(bp_model.layers[2].layers[i].get_weights())
    return model

def sample_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(z_log_var / 2) * eps

def conv_vae(image_shape, latent_dim, beta, quant_size=0, pruning='not_pruned'):

    int_size = QUANT_INT[quant_size]
    # encoder
    input_encoder = Input(shape=image_shape[1:], name='encoder_input')

    if quant_size!=0:
        quantized_inputs = QActivation('quantized_bits(16,10,0,alpha=1)')(input_encoder)
        x = ZeroPadding2D(((1,0),(0,0)))(quantized_inputs)
    else:
        quantized_inputs = None
        x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
    #
    x = BatchNormalization()(x)
    #
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    #
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    #
    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    #
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    #
    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Flatten()(x)
    #
    z_mean = Dense(latent_dim, name='latent_mu')(x) if quant_size==0 \
        else QDense(latent_dim, name='latent_mu',
               kernel_quantizer='quantized_bits(16,6,0,alpha=1)',
               bias_quantizer='quantized_bits(16,6,0,alpha=1)')(x)

    z_log_var = Dense(latent_dim, name='latent_sigma')(x) if quant_size==0 \
        else QDense(latent_dim, name='latent_sigma',
               kernel_quantizer='quantized_bits(16,6,0,alpha=1)',
               bias_quantizer='quantized_bits(16,6,0,alpha=1)')(x)

    z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

    encoder = Model(inputs=input_encoder, outputs=[z_mean, z_log_var, z], name='encoder_CNN')
    if pruning=='pruned':
        ''' How to estimate the enc step:
            num_images = input_train.shape[0] * (1 - validation_split)
            end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs
            start at 5: np.ceil(14508274/2*0.8/1024).astype(np.int32) * 5 = 28340
            stop at 15: np.ceil(14508274/2*0.8/1024).astype(np.int32) * 15 = 85020
        '''
        start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=start_pruning, end_step=end_pruning)
        encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
        encoder = encoder_pruned
    encoder.summary()

    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')

    x = Dense(64)(input_decoder)
    #
    x = Activation('relu')(x)
    #
    x = Reshape((2,1,32))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x)
    #
    x = Activation('relu')(x)
    #
    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((0,0),(1,1)))(x)
    #
    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x)
    x = Activation('relu')(x)
    #
    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((1,0),(0,0)))(x)
    #
    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x)
    #
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary()
    # vae
    vae_outputs = decoder(encoder(input_encoder)[2])
    vae = Model(input_encoder, vae_outputs, name='vae')
    vae.summary()
    # load weights
    if pruning=='pruned':
        vae = model_set_weights(vae, f'output/model-conv_vae-8-b0.8-q0-not_pruned', quant_size)
    # compile VAE
    vae.compile(optimizer=Adam(lr=3E-3, amsgrad=True),
                loss=make_mse_kl(z_mean, z_log_var, beta),
                metrics=[make_mse, make_kl(z_mean, z_log_var)]
                )
    return vae

def conv_ae(image_shape, latent_dim, quant_size=0, pruning='not_pruned'):
    int_size = QUANT_INT[quant_size]
    # encoder
    input_encoder = Input(shape=image_shape[1:], name='encoder_input')
    if quant_size!=0:
        quantized_inputs = QActivation(f'quantized_bits(16,10,0,alpha=1)')(input_encoder)
        x = ZeroPadding2D(((1,0),(0,0)))(quantized_inputs)
    else:
        quantized_inputs = None
        x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
    x = BatchNormalization()(x)
    #
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Flatten()(x)
    #
    enc = Dense(latent_dim)(x) if quant_size==0 \
        else QDense(latent_dim,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()
    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(64)(input_decoder) if quant_size==0 \
        else QDense(64,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(input_decoder)
    #
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    #
    x = Reshape((2,1,32))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((0,0),(1,1)))(x)

    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((1,0),(0,0)))(x)

    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(1, kernel_size=(3,3), use_bias=False, padding='same',
                        kernel_quantizer='quantized_bits(16,10,0,alpha=1)')(x)
    #
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary()

    if pruning=='pruned':
        start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=start_pruning, end_step=end_pruning)
        encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
        encoder = encoder_pruned
        decoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(decoder, pruning_schedule=pruning_schedule)
        decoder = decoder_pruned

    # ae
    ae_outputs = decoder(encoder(input_encoder))
    autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()
    # load weights
    if pruning=='pruned':
        autoencoder = model_set_weights(autoencoder, f'output/model-conv_ae-8-b0-q0-not_pruned', quant_size)
    # compile AE
    autoencoder.compile(optimizer=Adam(lr=3E-3, amsgrad=True),
        loss=make_mse)
    return autoencoder
