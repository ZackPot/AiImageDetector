import matplotlib.pyplot as plt
import numpy as np

fake_signals = np.load('preprocessing/fft_signals_fake.npy')
real_signals = np.load('preprocessing/fft_signals_real.npy')

fig, ax = plt.subplots(2, 1)
samples = 50

for i, signal in enumerate(fake_signals[0]):
    if (i % samples) == 0:
        ax[0].plot(signal)
    ax[0].title.set_text('A AI Image')

for i, signal in enumerate(real_signals[0]):
    if (i % samples) == 0:
        ax[1].plot(signal)
    ax[1].title.set_text('A Real Image')

fig.suptitle('FFT on Fake vs Real Images')
plt.show()