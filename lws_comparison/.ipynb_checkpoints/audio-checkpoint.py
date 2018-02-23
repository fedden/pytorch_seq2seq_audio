from model import to_var
import numpy as np
import librosa
import torch


def griffin_lim(spectrogram,
                n_iter=100,
                window='hann',
                n_fft=2048,
                hop_length=-1):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft,
                               hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return inverse


class AudioDataset():
    """A class to convert audio found at an url to magnitude frames."""

    def __init__(self,
                 fft_size,
                 hop_length,
                 batch_size,
                 sequence_length,
                 overlap_ratio=0.0,
                 lws_processor=None,
                 path=None,
                 limit=None):

        self.fft_size = fft_size
        self.feature_size = (fft_size // 2) + 1
        self.hop_length = hop_length

        self.data, self.sample_rate = librosa.load(path, mono=True)

        if limit is not None:
            self.data = self.data[:limit * self.sample_rate]

        # Get features to train on.
        if lws_processor is not None:
            self.stfts = lws_processor.stft(self.data)
            self.magnitudes = np.abs(self.stfts)

        else:
            self.stfts = librosa.stft(self.data,
                                      n_fft=fft_size,
                                      hop_length=hop_length)
            self.magnitudes, _ = librosa.magphase(self.stfts.T)

        self.dataset_size = len(self.magnitudes) - sequence_length * 2 - 1
        input_shape = (self.dataset_size, sequence_length, self.feature_size)
        target_shape = input_shape
        self.batch_size = batch_size
        self.x = np.zeros(input_shape, dtype=np.float32)
        self.y = np.zeros(target_shape, dtype=np.float32)
        
        if overlap_ratio < 0.0 or overlap_ratio > 1.0:
            message = '''Keep overlap percentage between zero and one.
            0.0 for no overlap at all; the model will predict entirely future frames.
            1.0 for 100% overlap; the model will predict the input (autoencoder).
            '''
            message  = 
            message += 'Zero for no overlap, one for '
            raise ValueError()

        for i, x_start in enumerate(range(0, self.dataset_size)):
            x_end = x_start + sequence_length
            y_start = x_end
            y_end = y_start + sequence_length

            self.x[i] = self.magnitudes[x_start:x_end]
            self.y[i] = self.magnitudes[y_start:y_end]

    def get_next_batch(self, amount_epochs):

        for epoch in range(amount_epochs):

            permuation = np.random.permutation(len(self.x))
            x, y = self.x[permuation], self.y[permuation]

            for start in range(0, len(x) - self.batch_size, self.batch_size):
                end = start + self.batch_size

                batch_x = torch.from_numpy(x[start:end])
                batch_y = torch.from_numpy(y[start:end])

                yield to_var(batch_x), to_var(batch_y), epoch
