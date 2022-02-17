from functions import make_mse_loss
import pathlib
import os
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as keras
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="total_val_loss")
        self.reconstruction_val_loss_tracker = keras.metrics.Mean(name="reconstruction_val_loss")
        self.kl_val_loss_tracker = keras.metrics.Mean(name="kl_val_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.total_val_loss_tracker,
            self.reconstruction_val_loss_tracker,
            self.kl_val_loss_tracker
        ]

    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data_in, training=True)
            reconstruction = self.decoder(z, training=True)
            
            reconstruction_loss = make_mse_loss(target, reconstruction) #one value
            beta = 0.8
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            kl_loss = tf.cast(kl_loss, tf.float32)
            total_loss = (1-beta)*reconstruction_loss + beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state((1-beta)*reconstruction_loss)
        self.kl_loss_tracker.update_state(beta*kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        #validation
        data_in, target = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = make_mse_loss(target, reconstruction) 
        beta=0.8
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss, axis=1)
        kl_loss = tf.cast(kl_loss, tf.float32)
        total_loss = (1-beta)*reconstruction_loss + beta*kl_loss
        self.total_val_loss_tracker.update_state(total_loss)
        self.reconstruction_val_loss_tracker.update_state((1-beta)*reconstruction_loss)
        self.kl_val_loss_tracker.update_state(beta*kl_loss)

        return {
            "loss": self.total_val_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_val_loss_tracker.result(),
            "kl_loss": self.kl_val_loss_tracker.result()
        }
    
    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print('saving model to {}'.format(path))
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path,'decoder.h5'))

    @classmethod
    def load(cls, path, custom_objects={}):
        ''' loading only for inference -> passing compile=False '''
        encoder = tf.keras.models.load_model(os.path.join(path,'encoder.h5'), custom_objects=custom_objects, compile=False)
        decoder = tf.keras.models.load_model(os.path.join(path,'decoder.h5'), custom_objects=custom_objects, compile=False)
        return encoder, decoder
    

class AE(Model):
    def __init__(self, autoenc, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.autoencoder = autoenc
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="total_val_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.total_val_loss_tracker
        ]

    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            reconstruction = self.autoencoder(data_in, training=True)
            total_loss = make_mse_loss(target, reconstruction)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

    def test_step(self, data):
        #validation
        data_in, target = data
        reconstruction = self.autoencoder(data_in)
        total_loss = make_mse_loss(target, reconstruction)
        self.total_val_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_val_loss_tracker.result()
        }
    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print('saving model to {}'.format(path))
        self.autoencoder.save(os.path.join(path, 'autoencoder.h5'))
    
    @classmethod
    def load(cls, path, custom_objects={}):
        ''' loading only for inference -> passing compile=False '''
        autoencoder = tf.keras.models.load_model(os.path.join(path,'autoencoder.h5'), custom_objects=custom_objects, compile=False)
        return autoencoder