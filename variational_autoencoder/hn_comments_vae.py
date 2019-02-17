"""Example of VAE on Hacker News Comments dataset

From: https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/7208611eef764273bf5ece01d8ee83b33d73d448/chapter8-vae/vae-cnn-mnist-8.1.2.py

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate samples by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.



# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114

# Run
Takes about 10m to run on Samsung (1GPU)
Requires no inputs

"""

from keras.layers import Dense, Input, Embedding
from keras.layers import Flatten, Lambda
from keras.models import Model
from keras.initializers import Constant
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
from datetime import datetime

import argparse
import h5py
import numpy as np


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Reference:
        https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))  # This is not a layer
    return z_mean + K.exp(0.5 * z_log_var) * epsilon  # This does become a layer


def test_gpu():
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


###
# Sanity checks to try to prevent freezing
# Last test: After restart, running this from CLI caused one CPU core to rail, but didn't freeze computer
# No amout of killing processes seemed to help - they wouldn't die
###
K.clear_session()
tf.reset_default_graph()  # for being sure
test_gpu()


def main():
    """ Wrapped in a function to encourage garbage collection
    
    Hopefully, this will prevent my laptop from freezing as much
    """
    with h5py.File('data/indexed.h5', 'r') as f_indexed:
        x_train = f_indexed['training_data'][()]
        sequence_length = x_train.shape[1]

    with h5py.File('data/w2v_weights_and_word_index_mapping.h5', 'r') as f_w2v:
        embedding_matrix = f_w2v['weights'][()]
        embedding_matrix_scaled = (embedding_matrix - np.min(embedding_matrix)) / \
                                  (np.max(embedding_matrix) - np.min(embedding_matrix))
        num_distinct_words, embedding_dim = embedding_matrix_scaled.shape

    # network parameters
    input_shape = (sequence_length,)
    latent_dim = 2  # The number of dimensions to project onto
    batch_size = 128
    epochs = 10

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')

    # This is the most efficient way to include the w2v
    # Doing the joins outside of the GPU is impractical
    embedded_sequences = Embedding(num_distinct_words,
                                   embedding_dim,
                                   embeddings_initializer=Constant(embedding_matrix_scaled),
                                   input_length=sequence_length,
                                   trainable=False)(inputs)

    # I don't want to deal with CNN right now
    # I'm not sure if it's possible, since it would be outside of the
    # optimization loop. I guess you could use Conv1DTranspose
    post_cnn = embedded_sequences

    # generate latent vector Q(z|X)
    flattened = Flatten(name='flattened')(post_cnn)  # Because this will be compared to a sigmoid, ensure range 0-1
    dense1_enc = Dense(128, activation='relu', name='dense1_enc')(flattened)
    dense2_enc = Dense(32, activation='relu', name='dense2_enc')(dense1_enc)
    z_mean = Dense(latent_dim, name='z_mean')(dense2_enc)
    z_log_var = Dense(latent_dim, name='z_log_var')(dense2_enc)

    # use reparameterization trickshape to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    plot_model(encoder, to_file='images/hn_vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    dense2_dec = Dense(32, activation='relu', name='dense2_dec')(latent_inputs)
    dense1_dec = Dense(128, activation='relu', name='dense1_dec')(dense2_dec)
    outputs_dec = Dense(K.int_shape(flattened)[1], activation='sigmoid')(dense1_dec)  # Connection to Enc

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs_dec, name='decoder')
    plot_model(decoder, to_file='images/hn_vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs_vae = decoder(encoder(inputs)[2])  # This can't just be 'outputs_dec', not sure why
    vae = Model(inputs, outputs_vae, name='vae')

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    help_ = "Actually train the model (Remember to save: Computer may freeze)"
    parser.add_argument("-t", "--train", help=help_, action='store_true')
    args = parser.parse_args()

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(flattened), K.flatten(outputs_vae))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(flattened),
                                                  K.flatten(outputs_vae))

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file='images/hn_vae_cnn.png', show_shapes=True)

    # Set up tensorboard
    t = datetime.now()
    ld = '/home/mritter/code/twitter_nlp/full_hn_usernames/tensorboard_logs/{:%Y-%m-%d-%H-%M}'.format(t)
    tensorboard = TensorBoard(log_dir=ld)

    if args.train:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tensorboard])
        vae.save_weights('data/hn_vae_cnn_weights{:%Y-%m-%d-%H-%M}.h5'.format(t))


if __name__ == '__main__':
    main()
