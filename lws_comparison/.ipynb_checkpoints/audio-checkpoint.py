from model import to_var
import numpy as np
import librosa
import torch
import lws


def polar_to_cartesian(magnitude, theta):
    """Polar to cartesian coordinates."""
    return magnitude * np.exp(1j * theta)


def phase_angle_inc_generator(fft_size, hop_size, sample_rate):
    """Returns the ammount we are going to increment every bin's 
    phase value by.
    """
    # Finding the period of each bin's central frequency of my fft.
    fft_frequencies = librosa.fft_frequencies(sr=sample_rate,
                                              n_fft=fft_size)
    fft_frequencies[0] = 1
    fft_freq_period_sample = sample_rate / fft_frequencies
    
    # Stopping the 0's element of fft_freq_period_sample from being "inf".
    # MAJOR ISSUE: 0th element is inf, 1st element is 2048, 1024th element 
    # is 2. What is the 0th element and why does this STFT return 1 more 
    # value than half the frameSize.
    # The answer is probably the reason the phase vocoding sounds lame.
    # Going to make the 0th bin phase 0 later in this code.
    fft_freq_period_sample[0] = fft_size

    # Divide the hop length by the period of the bins
    # to create an value to increment the phase by for each hop.
    # and scaling the the value to a number between 2*PI.
    hop_increments = (hop_size / fft_freq_period_sample) * 2 * np.pi
    hop_increments[0] = 0.0
    return hop_increments


def vocoder_phases(amount_magnitude_frames, fft_size, hop_size, sample_rate):
    """This function generates a frame of phases for each mag frame.
    amount_magnitude_frames denotes the amount of magnitude values.
    """
    phase_angle_inc = phase_angle_inc_generator(fft_size, 
                                                hop_size, 
                                                sample_rate)
    
    # Initialize a random array of the number of phases in each set with 
    # a value between 0 and 2 * Pi. If we don't start with with random 
    # values then we get a sweeping effect when they all start from 0 at 
    # the begining of the audio. This array will be incrememted in the 
    # loop below.
    current_phase = np.random.rand(int(fft_size/2)+1) * 2*np.pi
    current_phase[0] = 0.0
    
    new_phases = np.zeros((amount_magnitude_frames, int(fft_size/2)+1))
    
    for i in range(amount_magnitude_frames):
        new_phases[i] = current_phase
        # Increment current phase with the increment values.
        current_phase = current_phase + phase_angle_inc
        # Modulo the incremented phase by 2*pi to make sure it stays in 
        # the range of phases.
        current_phase = current_phase % 2*np.pi
    
    # Return the phase values so they are a range of -Pi to Pi.
    return new_phases - np.pi


def griffin_lim(spectrogram, n_iter=100, window='hann', n_fft=2048, hop_length=-1):
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


def overlap_and_add(blocks, block_step, window_fn=np.hanning, overlap_axis=1):
    '''Thanks Memo!
    Can operate on any rank ndarray, [n, t, ...] where t is the overlap axis'''
    block_step = int(block_step)
    window_length = blocks.shape[overlap_axis]
    if window_fn:
        window = window_fn(window_length)
        dim_array = np.ones((1, blocks.ndim), int).ravel()
        dim_array[overlap_axis] = -1
        window = window.reshape(dim_array)
        blocks = blocks * window
     
    shape0 = (len(blocks)-1) * block_step + window_length
    signal = np.zeros(shape=(shape0,)+blocks.shape[2:], dtype=blocks.dtype.type)
    for i in range(0, len(blocks)):
        signal[i*block_step:i*block_step+window_length,...] += blocks[i,...] 
    signal /= 0.5 * window_length / block_step
     
    return signal


def magnitudes_to_audio(magnitudes, settings, dataset, phase_estimation_type):
    # Overlap results if required.
    if settings.overlap_ratio > 0.0:
        new_shape = (-1, settings.sequence_length, settings.feature_size)
        magnitudes = magnitudes.reshape(new_shape)
        magnitudes = overlap_and_add(magnitudes, dataset.offset)
            
    # Reconstruct phase and generate samples.
    # Did we specify griffin lim for phase reconstruction?
    if phase_estimation_type == 'griffin_lim':
        predicted_audio = griffin_lim(magnitudes.T,
                                      n_iter=settings.griffin_lim_iterations,
                                      window='hann',
                                      n_fft=settings.fft_size,
                                      hop_length=settings.hop_length)
    
    # Did we specify lws for phase reconstruction?
    elif phase_estimation_type == 'lws':
            
        # Reconstruct phases.
        double_mags = magnitudes.astype(np.float64)
        predicted_stfts = dataset.lws_processor.run_lws(double_mags)
        
        # Did we use the librosa or lws stft originally to get the mags?
        if settings.lws_mags:
            predicted_audio = dataset.lws_processor.istft(predicted_stfts)
        else:
            predicted_audio = librosa.istft(predicted_stfts.T, 
                                            hop_length=settings.hop_length)
        
    # Did we specify vocoder-based phase reconstruction?
    elif phase_estimation_type == 'vocoder':
        phases = vocoder_phases(magnitudes.shape[0], 
                                settings.fft_size, 
                                settings.hop_length, 
                                dataset.sample_rate)
        predicted_stfts = polar_to_cartesian(magnitudes, phases)
        
        # Did we use the librosa or lws stft originally to get the mags?
        if settings.lws_mags:
            predicted_audio = dataset.lws_processor.istft(predicted_stfts)
        else:
            predicted_audio = librosa.istft(predicted_stfts.T, 
                                            hop_length=settings.hop_length)
        
    return predicted_audio
    

class AudioDataset():
    """A class to convert audio found at an url to magnitude frames."""

    def __init__(self, settings, limit=None):

        self.fft_size = settings.fft_size
        self.feature_size = (settings.fft_size // 2) + 1
        self.hop_length = settings.hop_length
        self.base_epoch = settings.epoch

        self.data, self.sample_rate = librosa.load(settings.path, mono=True)
        if limit is not None:
            self.data = self.data[:limit * self.sample_rate]
            
        self.lws_processor = lws.lws(settings.fft_size,
                                     settings.hop_length,
                                     mode=settings.mode,
                                     perfectrec=settings.perfect_reconstruction) 

        # Get features to train on.
        if settings.lws_mags:
            self.stfts = self.lws_processor.stft(self.data)
            self.magnitudes = np.abs(self.stfts)

        else:
            self.stfts = librosa.stft(self.data,
                                      n_fft=settings.fft_size,
                                      hop_length=settings.hop_length)
            self.magnitudes, _ = librosa.magphase(self.stfts.T)

        self.dataset_size = len(self.magnitudes) - settings.sequence_length * 2 - 1
        input_shape = (self.dataset_size, settings.sequence_length, self.feature_size)
        target_shape = input_shape
        self.batch_size = settings.batch_size
        self.x = np.zeros(input_shape, dtype=np.float32)
        self.y = np.zeros(target_shape, dtype=np.float32)
        
        if settings.overlap_ratio < 0.0 or settings.overlap_ratio > 1.0:
            message = '''Keep overlap percentage between zero and one.
            0.0 for no overlap at all; the model will predict entirely future frames.
            1.0 for 100% overlap; the model will predict the input (autoencoder).
            '''
            raise ValueError(message)
            
        self.offset = int(settings.sequence_length * settings.overlap_ratio)

        for i, x_start in enumerate(range(0, self.dataset_size)):
            x_end = x_start + settings.sequence_length
            y_start = x_end - self.offset
            y_end = y_start + settings.sequence_length

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

                yield to_var(batch_x), to_var(batch_y), epoch + self.base_epoch
