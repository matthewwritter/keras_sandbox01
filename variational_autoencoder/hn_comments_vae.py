'''Example of VAE on MNIST dataset using CNN

From: https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/7208611eef764273bf5ece01d8ee83b33d73d448/chapter8-vae/vae-cnn-mnist-8.1.2.py

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114

# Run
Takes about 10m to run on Samsung (1GPU)
Requires no inputs
After first run, re-do generative ability with `%run sandbox.py -w vae_cnn_mnist.h5`
(make sure to enable %matplotlib inline)

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input, Embedding
from keras.layers import Flatten, Lambda
from keras.models import Model
from keras.initializers import Constant
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import argparse
import h5py


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

###
# Sanity checks to prevent freezing
###
K.clear_session()

# IF IT DOES NOT WORK, MAY NEED TO RESTART COMPUTER

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

# confirm PyTorch sees the GPU
from torch import cuda
assert cuda.is_available()
assert cuda.device_count() > 0
print(cuda.get_device_name(cuda.current_device()))

def main():
    with h5py.File('data/indexed.h5', 'r') as f_indexed:
        x_train = f_indexed['training_data'][()]
        sequence_length = x_train.shape[1]

    with h5py.File('data/w2v_weights_and_word_index_mapping.h5', 'r') as f_w2v:
        embedding_matrix = f_w2v['weights'][()]
        num_distinct_words, embedding_dim = embedding_matrix.shape

    # network parameters
    input_shape = (sequence_length, )
    latent_dim = 2  # The number of dimensions to project onto
    batch_size = 32  # TODO Update once working
    epochs = 2  # TODO Update once working


    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')

    # This is the most efficient way to include the w2v
    # Doing the joins outside of the GPU is impractical
    embedded_sequences = Embedding(num_distinct_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=sequence_length,
                                trainable=False)(inputs)

    # I don't want to deal with CNN right now
    # I'm not sure if it's possible, since it would be outside of the
    # optimization loop. I guess you could use Conv1DTranspose
    post_cnn = embedded_sequences

    # generate latent vector Q(z|X)
    flattened = Flatten(name='flattened')(post_cnn)
    dense1 = Dense(16, activation='relu', name='dense1')(flattened)
    z_mean = Dense(latent_dim, name='z_mean')(dense1)
    z_log_var = Dense(latent_dim, name='z_log_var')(dense1)

    # use reparameterization trickshape to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    outputs = Dense(K.int_shape(flattened)[1], activation='sigmoid')(latent_inputs)  # TODO probably add more layers

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(flattened), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(flattened),
                                                  K.flatten(outputs))

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    from keras.callbacks import TensorBoard as tb
    from datetime import datetime
    t = datetime.now()
    tensorboard = tb(log_dir='/home/mritter/code/twitter_nlp/full_hn_usernames/tensorboard_logs/{:%Y-%m-%d-%H-%M}'.format(t))

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tensorboard])
        vae.save_weights('vae_cnn_mnist.h5')


if __name__ == '__main__':
    main()
