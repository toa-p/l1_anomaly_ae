import tensorflow as tf
from tensorflow.python.keras.layers.merge import _Merge as Merge
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import math

class Distance(Merge):
    def _check_inputs(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `{}` layer should be called '
                             'on exactly 2 inputs'.format(self.__class__.__name__))
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super(Distance, self).build(input_shape)
        self._check_inputs(input_shape)
        
class KLLoss(Distance):
    def _merge_function(self, inputs):
        self._check_inputs(inputs)
        mean = inputs[0]
        log_var = inputs[1]
        kl = 1. + log_var - math_ops.square(mean) - math_ops.exp(log_var)
        kl = -0.5 * math_ops.reduce_mean(kl, axis=-1, keepdims=True)
        return kl
    
class Radius(Distance):
    def _merge_function(self, inputs):
        self._check_inputs(inputs)
        mean = inputs[0]
        log_var = inputs[1]
        sigma = math_ops.exp(log_var)
        radius = math_ops.div_no_nan(math_ops.square(mean), sigma)
        radius = math_ops.reduce_sum(radius, axis=-1, keepdims=True)
        return radius 
    
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class CustomMSE(Distance):

    def _merge_function(self, inputs):
        self._check_inputs(inputs)

        true = inputs[0]
        predicted = inputs[1]
        # remove last dimension
        true = tf.squeeze(true, axis=-1)
        true = tf.cast(true, dtype=tf.float32)
        # trick with phi
        outputs_phi = math.pi*math_ops.tanh(predicted)
        # trick with phi
        outputs_eta_egamma = 3.0*math_ops.tanh(predicted)
        outputs_eta_muons = 2.1*math_ops.tanh(predicted)
        outputs_eta_jets = 4.0*math_ops.tanh(predicted)
        outputs_eta = tf.concat([predicted[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
        # use both tricks
        predicted = tf.concat([predicted[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
        # mask zero features
        mask = math_ops.not_equal(true,0)
        mask = tf.cast(mask, tf.float32)
        predicted = mask * predicted

        true = tf.reshape(true, [-1, 57])
        predicted = tf.reshape(predicted, [-1, 57])

        return  math_ops.reduce_mean(math_ops.square(true-predicted), axis=-1, keepdims=True)
