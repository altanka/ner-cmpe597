from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Lambda, Input, Dense, LSTM, RepeatVector
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])
# normalize the pixel value:
x_train = x_train / 255
x_test = x_test / 255


def lstm_vae():
    latent_dim = 2
    timesteps = 28
    input_dim = 28
    intermediate_dim = 64

    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    mean = Dense(latent_dim)(h)
    log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        mean, log_sigma = args
        batch_size = K.shape(mean)[0]

        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=1.)
        return mean + log_sigma * epsilon

    # Z Layer - VAE
    z = Lambda(sampling)([mean, log_sigma])

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # Autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder for input to latent space
    encoder = Model(x, mean)

    decoder_input = Input(shape=(latent_dim,))

    h_decoded_generator = RepeatVector(timesteps)(decoder_input)
    h_decoded_generator = decoder_h(h_decoded_generator)

    x_decoded_mean_generator = decoder_mean(h_decoded_generator)
    # generator for reconstructing inputs
    generator = Model(decoder_input, x_decoded_mean_generator)

    reconstruction_loss = binary_crossentropy(x, x_decoded_mean)*28*28

    # KL divergence
    kl_loss = - 0.5 * K.mean(1 + log_sigma -
                             K.square(mean) - K.exp(log_sigma))

    # Total loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')

    return encoder, generator, vae


if __name__ == "__main__":
    encoder, generator, vae = lstm_vae()
    vae.summary()
    vae.fit(x_train, epochs=50, batch_size=128, validation_data=(x_test, None))
    vae.save_weights('vae_weights.npy')
    history = vae.history

    plt.figure(facecolor='white')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('vae_loss.png')
