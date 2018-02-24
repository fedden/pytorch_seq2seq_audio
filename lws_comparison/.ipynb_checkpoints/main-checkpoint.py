from __future__ import print_function
from __future__ import division

import os
import lws
import torch
import librosa
import argparse
import numpy as np
from model import to_var
from settings import Settings
from utils import get_file_name
from audio import vocoder_phases, polar_to_cartesian
from audio import AudioDataset, griffin_lim, overlap_and_add
from neural_methods import get_model_and_optimisers, train_batch
from neural_methods import train, inference, load_model, save_model

# All the arguments one can set for this program.
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True)
parser.add_argument('--sanity_check', action='store_true')
parser.add_argument('--audio_path', required=True)
parser.add_argument('--load_path')
parser.add_argument('--save_path')
parser.add_argument('--overlap_ratio')
parser.add_argument('--fft_size', required=True)
parser.add_argument('--lws_mags', required=True)
parser.add_argument('--phase_estimation_method', required=True)
parser.add_argument('--griffin_lim_iterations')
parser.add_argument('--perfect_reconstruction')
parser.add_argument('--lws_mode')
parser.add_argument('--cuda_device')
parser.add_argument('--encoder_layers')
parser.add_argument('--decoder_layers')
parser.add_argument('--dropout')
parser.add_argument('--inference_length')
parser.add_argument('--learning_rate')
args = vars(parser.parse_args())

# Houses the settings.
settings = Settings(args)

# Only use specified GPU.
with torch.cuda.device(settings.cuda_device):       

    # Get model and acillary components.
    encoder, decoder, criterion, encoder_optimiser, decoder_optimiser = \
        get_model_and_optimisers(settings)
        
    # Load previous weights if necessary.
    if settings.load_path is not None:
        encoder, decoder = load_model(settings, encoder, decoder)
        
    # Potentially construct lws processor.
    lws_processor = None
    if settings.lws_mags:
        lws_processor = lws.lws(settings.fft_size,
                                settings.hop_length,
                                mode=settings.mode,
                                perfectrec=settings.perfect_reconstruction) 

    # Create audio dataset. Potentially use lws processor.
    dataset = AudioDataset(settings, lws_processor=lws_processor)
    
    # What we are training to build.
    file_name = get_file_name(settings)
    print("training and ultimately producing:", file_name)

    # Train model.
    train(encoder, decoder, 
          encoder_optimiser, 
          decoder_optimiser, 
          dataset, settings,
          criterion)
            
    # Save if necessary.
    if settings.save_path is not None:
        save_model(settings, encoder, decoder)

    # Inference, first get random starting points for the model.
    indices = np.random.permutation(dataset.dataset_size)[:settings.batch_size]
    start_sequences = to_var(torch.from_numpy(dataset.x[indices]))
    
    # Run inference.
    predicted_batch = inference(encoder, 
                                decoder, 
                                start_sequences, 
                                settings.inference_length, 
                                init_hidden_once=True)
    first_magnitudes = predicted_batch[0]
    
    # Overlap results if required.
    if settings.overlap_ratio > 0.0:
        new_shape = (-1, settings.sequence_length, settings.feature_size)
        first_magnitudes = first_magnitudes.reshape(new_shape)
        first_magnitudes = overlap_and_add(first_magnitudes, dataset.offset)
            
    # Reconstruct phase and generate samples.
    # Did we specify griffin lim for phase reconstruction?
    if settings.griffin_lim_phase:
        predicted_audio = griffin_lim(first_magnitudes,
                                      n_iter=settings.griffin_lim_iterations,
                                      window='hann',
                                      n_fft=settings.fft_size,
                                      hop_length=settings.hop_length)
    
    # Did we specify lws for phase reconstruction?
    elif settings.lws_phase:
        
        # If we previously used librosa stft.
        if lws_processor is None:
            lws_processor = lws.lws(settings.fft_size,
                                    settings.hop_length,
                                    mode=settings.mode,
                                    perfectrec=settings.perfect_reconstruction)
            
        # Reconstruct phases.
        predicted_stfts = lws_processor.run_lws(first_magnitudes)
        predicted_audio = lws_processor.istft(predicted_stfts)
        
    # Did we specify vocoder-based phase reconstruction?
    elif settings.vocoder_phase:
        phases = vocoder_phases(first_magnitudes.shape[0], 
                                settings.fft_size, 
                                settings.hop_length, 
                                dataset.sample_rate)
        real = polar_to_cartesian(first_magnitudes, phases)
        predicted_audio = librosa.istft(real.T, hop_length=settings.hop_length)

    # Get file name and write to wav.
    librosa.output.write_wav(file_name, predicted_audio, dataset.sample_rate)
    print("Done. Thanks for using and exiting.")
