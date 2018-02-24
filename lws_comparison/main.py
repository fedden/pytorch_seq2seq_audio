from __future__ import print_function
from __future__ import division

import os
import torch
import librosa
import argparse
import numpy as np
from model import to_var
from settings import Settings
from utils import get_file_name
from audio import AudioDataset, magnitudes_to_audio
from neural_methods import get_model_and_optimisers, train_batch
from neural_methods import train, inference, load_model, save_model

# All the arguments one can set for this program.
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True)
parser.add_argument('--sanity_check', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--audio_path', required=True)
parser.add_argument('--load_path')
parser.add_argument('--save_path')
parser.add_argument('--overlap_ratio')
parser.add_argument('--fft_size', required=True)
parser.add_argument('--use_lws_mags', required=True)
parser.add_argument('--phase_estimation_methods', nargs='+', required=True)
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

    # Get model and ancillary learning components.
    encoder, decoder, criterion, encoder_optimiser, decoder_optimiser = \
        get_model_and_optimisers(settings)
        
    # Load previous weights if necessary.
    if settings.load_path is not None:
        encoder, decoder = load_model(settings, encoder, decoder)

    # Create audio dataset. Potentially use lws processor.
    dataset = AudioDataset(settings)

    # Train model.
    train(encoder, decoder, 
          encoder_optimiser, 
          decoder_optimiser, 
          dataset, settings,
          criterion)
    
    if settings.verbose:
        print('Finished training.')
            
    # Save if necessary.
    if settings.save_path is not None:
        if settings.verbose:
            print('Saving model in {}'.format(settings.save_path))
        
        save_model(settings, encoder, decoder)

    # Inference, first get random starting points for the model.
    indices = np.random.permutation(dataset.dataset_size)[:settings.batch_size]
    start_sequences = to_var(torch.from_numpy(dataset.x[indices]))
    
    if settings.verbose:
        print('Starting inference.')
    
    # Run inference and get an array of magnitudes.
    predicted_batch = inference(encoder, 
                                decoder, 
                                start_sequences, 
                                settings.inference_length, 
                                init_hidden_once=True)
    magnitudes = predicted_batch[0]
    
    if settings.verbose:
        print('Finished inference. Beginning phase estimation method(s).')
    
    # Create audio with the phase estimation types priorly specified.
    for phase_estimation_method in settings.phase_estimation_methods:
        
        # Get audio from mags using phase estimation method.
        predicted_audio = magnitudes_to_audio(magnitudes, 
                                              settings, 
                                              dataset, 
                                              phase_estimation_method)
        
        # Adjust file name with respective phase method and write to wav.
        save_name = get_file_name(settings)
        save_name += '_phase_method_{}'.format(phase_estimation_method)
        save_name += '.wav'
        
        # If specified save path, save output there. Else save in working directory.
        if settings.save_path is not None:
            file_name = os.path.join(settings.save_path, save_name)
            
        librosa.output.write_wav(file_name, predicted_audio, dataset.sample_rate)
        
        if settings.verbose:
            print("Saved {} to disk.".format(file_name))
        
    if settings.verbose:
        print("Done. Thanks for using. Now exiting.")
