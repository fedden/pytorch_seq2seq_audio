import numpy as np
import librosa
from utils import to_var, random_mask
import torch
import io
import soundfile as sf
from six.moves.urllib.request import urlopen


class AudioDataset():
    """A class to convert audio found at an url to magnitude frames."""

    def __init__(self,
                 fft_size,
                 hop_length,
                 batch_size,
                 sequence_length,
                 url=None,
                 path=None,
                 limit=None,
                 input_noise=0.0):

        # Sanity checks.
        data_str = "One (and only one) of these should be passed."
        assert (url is not None) or (path is not None), data_str
        if url:
            assert path is None, data_str
        elif path:
            assert url is None, data_str

        self.fft_size = fft_size
        self.feature_size = (fft_size // 2) + 1
        self.hop_length = hop_length

        # Get data.
        if url:
            self.data, self.sample_rate = \
                sf.read(io.BytesIO(urlopen(url).read()))
        else:
            self.data, self.sample_rate = librosa.load(path)

        # To mono and trim if necessary.
        self.data = np.mean(self.data, axis=1)
        if limit is not None:
            self.data = self.data[:limit * self.sample_rate]

        # Get features to train on.
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

        for i, x_start in enumerate(range(0, self.dataset_size)):
            y_start = x_end = x_start + sequence_length
            y_end = y_start + sequence_length

            self.x[i] = self.magnitudes[x_start:x_end]
            self.y[i] = self.magnitudes[y_start:y_end]

    def get_next_batch(self, amount_epochs):

        for epoch in range(amount_epochs):

            permuation = np.random.permutation(len(self.x))
            x, y = self.x[permuation], self.y[permuation]

            for start in range(0, len(x) - self.batch_size, self.batch_size):
                end = start + self.batch_size

                batch_x = random_mask(x[start:end], self.input_noise)
                batch_y = y[start:end]

                batch_x = torch.from_numpy(batch_x)
                batch_y = torch.from_numpy(batch_y)

                yield to_var(batch_x), to_var(batch_y), epoch
