import numpy as np
import matplotlib.pyplot as plt

from vae_lstm_train import lstm_vae
import matplotlib
matplotlib.use('Agg')


def save_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :].reshape(28, 28, 1), cmap='gray_r')
    plt.show()
    plt.savefig('vae_generated_images.png')


encoder, generator, vae = lstm_vae()
vae.summary()
vae.load_weights('vae_weights.npy')
try:
    f = open('inputs.npy', 'rb')
    latent_points = np.load(f)
    X = generator.predict(latent_points)
    save_plot(X, 10)
except:
    print('Please run gan generator first')
