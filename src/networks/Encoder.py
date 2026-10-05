import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Encoder(tf.keras.Model):
    def __init__(self, n_neurons, latent_dim, activation='relu'):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(l, activation=activation) for l in n_neurons]
        out_dim = tfp.layers.MultivariateNormalTriL.params_size(latent_dim)
        self.out = tf.keras.layers.Dense(out_dim, activation=None)

        self.dist = tfp.layers.MultivariateNormalTriL(latent_dim)

    def call(self, x):
        z = x
        for layer in self.hidden:
            z = layer(z)
        z = self.out(z)
        return self.dist(z)

    def init_output_layer(self, bias, weight_scale=1e-4):
        """
        Initializes the output layer of the encoder, 
        according to the prior distribution parameters in the referenced paper.
        """
        weights, _ = self.out.get_weights()
        weights = weights * weight_scale
        new_bias = np.array(bias, dtype=np.float32)
        self.out.set_weights([weights, new_bias])