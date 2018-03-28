import lws
import librosa
import random
import numpy as np
import math


def _random_occlusion(mags, min_amount, max_amount):
    
    width, height = mags.shape
    
    def random_from_exp(amount_min, amount_max, exp): 
        return random.uniform(amount_min, math.sqrt(amount_max))**exp
    
    min_width = int(min_amount * width)
    max_width = int(max_amount * width)
    min_height = int(min_amount * height)
    max_height = int(max_amount * height)
    
    noise_width = np.random.randint(min_width, max_width)
    noise_height = np.random.randint(min_height, max_height)
    
    x_start = np.random.randint(width - noise_width)
    y_start = int((random_from_exp(0, 10, 2) / 10) * (height - noise_height))
    
    x_end = x_start + noise_width
    y_end = y_start + noise_height

    shape = mags[x_start:x_end:, y_start:y_end:].shape
    
    noise = np.random.normal(loc=np.mean(mags), 
                             scale=np.std(mags), 
                             size=shape)
    mags[x_start:x_end:, y_start:y_end:] = noise
    return mags


def _multiplacative_noise(mags, high=1.0):
    high = np.random.random_sample() * high
    noise = np.random.uniform(high=high, size=mags.shape)
    return noise.astype(mags.dtype) * mags


def _additive_noise(mags, high=1.0):
    high = np.random.random_sample() * high
    noise = np.random.uniform(high=high, size=mags.shape)
    return noise.astype(mags.dtype) + mags


def _random_masking(mags, masking_prob):
    masking_prob = np.random.random_sample() * masking_prob
    mask = np.random.choice([0, 1], size=mags.shape, 
                            p=[masking_prob, 1.0 - masking_prob])
    return mask.astype(mags.dtype) * mags


class OnlineAudioAugmentor():
    
    def __init__(self):
        self.prob_random_occlusion = 0
        self.prob_additive_noise = 0
        self.prob_multiplacative_noise = 0
        self.prob_random_masking = 0
    
    def random_occlusion(self, prob, min_amount, max_amount):
        self.prob_random_occlusion = prob
        self.min_amount_random_occlusion = min_amount
        self.max_amount_random_occlusion = max_amount

    def additive_noise(self, prob, high):
        self.prob_additive_noise = prob
        self.high_additive_noise = high
    
    def multiplacative_noise(self, prob, high):
        self.prob_multiplacative_noise = prob
        self.high_multiplacative_noise = high
    
    def random_masking(self, prob, masking_prob):
        self.prob_random_masking = prob
        self.masking_prob_random_masking = masking_prob
        
    def augment(self, mags):
        
        batch_size = len(mags)
        
        if mags.ndim != 3:
            raise ValueError('augment expects a batch of magnitudes with a dimensionality of 3.')
            
        for i in range(batch_size):
            if np.random.random_sample() < self.prob_multiplacative_noise:
                mags[i] = _multiplacative_noise(mags[i], self.high_multiplacative_noise)

            if np.random.random_sample() < self.prob_random_occlusion:
                mags[i] = _random_occlusion(mags[i], 
                                            self.min_amount_random_occlusion,
                                            self.max_amount_random_occlusion)

            if np.random.random_sample() < self.prob_additive_noise:
                mags[i] = _additive_noise(mags[i], self.high_additive_noise)

            if np.random.random_sample() < self.prob_random_masking:
                mags[i] = _random_masking(mags[i], self.masking_prob_random_masking)
            
        return mags