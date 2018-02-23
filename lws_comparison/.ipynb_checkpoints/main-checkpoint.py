from __future__ import print_function

import os
import lws
import librosa
import argparse
import numpy as np
import torch
import torch.nn as nn
from neural_methods import train, run
from utils import get_file_name, str_to_bool
from model import EncoderRNN, AttentionDecoderRNN, to_var
from audio import AudioDataset, griffin_lim, overlap_and_add

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=True)
parser.add_argument('--load_path')
parser.add_argument('--save_path')
parser.add_argument('--overlap_ratio')
parser.add_argument('--fft_size', required=True)
parser.add_argument('--lws_mags', required=True)
parser.add_argument('--griffin_lim_phase', required=True)
parser.add_argument('--griffin_lim_iterations')
parser.add_argument('--perfect_reconstruction')
parser.add_argument('--lws_mode')
parser.add_argument('--cuda_device')
args = vars(parser.parse_args())

fft_size = int(args['fft_size'])
lws_mags = str_to_bool(args['lws_mags'])

griffin_lim_phase = str_to_bool(args['griffin_lim_phase'])
griffin_lim_iterations = int(args['griffin_lim_iterations']) if args['griffin_lim_iterations'] != None else None
perfect_reconstruction = str_to_bool(args['perfect_reconstruction'])
mode = args['lws_mode']
cuda_device = 0 if args['cuda_device'] == None else int(args['cuda_device'])
sequence_length = 150
batch_size = 8
overlap_ratio = 0.0 if args['overlap_ratio'] == None else float(args['overlap_ratio'])
save_path = str(args['save_path']) if args['save_path'] else None
load_path = str(args['load_path']) if args['load_path'] else None
number_epochs = int(args['epochs'])
feature_size = input_size = hidden_size = (fft_size // 2) + 1
hop_length = fft_size // 4
path = "../notebooks/massive_chops/trimmed/vocals_trimmed.wav"

if not griffin_lim_phase and (mode is None or perfect_reconstruction is None):
    raise ValueError('mode and perfect reconstruction parameters must both be set.')

if griffin_lim_phase and (griffin_lim_iterations is None):
    raise ValueError('you must set a value for the griffin lim iterations - 100 is a good value.')

file_name = get_file_name(fft_size, lws_mags, griffin_lim_phase,
                          griffin_lim_iterations, perfect_reconstruction,
                          mode, overlap_ratio)
print("training and ultimately producing:", file_name)

with torch.cuda.device(cuda_device):

    if lws_mags:
        lws_processor = lws.lws(fft_size,
                                hop_length,
                                mode=mode,
                                perfectrec=perfect_reconstruction)
    else:
        lws_processor = None

    # Construct RNNs and optimisers.
    encoder = EncoderRNN(input_size=feature_size,
                         batch_size=batch_size,
                         hidden_size=feature_size,
                         number_layers=3,
                         dropout=0.0)

    decoder = AttentionDecoderRNN('general',
                                  batch_size=batch_size,
                                  hidden_size=hidden_size,
                                  output_size=feature_size,
                                  number_layers=3,
                                  dropout=0.0)

    criterion = nn.MSELoss()

    # Enable GPU tensors provided GPUs actually exist!
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=0.00001)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=0.00005)

    dataset = AudioDataset(fft_size=fft_size,
                           hop_length=hop_length,
                           batch_size=batch_size,
                           lws_processor=lws_processor,
                           sequence_length=sequence_length,
                           overlap_ratio=overlap_ratio,
                           path=path)
    
    # Load.
    if load_path != None:
        encoder_path = os.path.join(load_path, 'encoder_model.pytorch')
        decoder_path = os.path.join(load_path, 'decoder_model.pytorch')
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

    # Train.
    previous_epoch = 0
    for x, y, epoch in dataset.get_next_batch(number_epochs):

        loss = train(encoder, decoder, encoder_optimiser, decoder_optimiser,
                     x, y, criterion, 5.0)

        if epoch != previous_epoch:
            previous_epoch = epoch
            print('')
        else:
            epoch_str = "epoch {}/{}, loss {}     "
            print(epoch_str.format(epoch, number_epochs, loss), end="\r")
            
    # Save.
    if save_path != None:
        
        folder_name = 'epoch_{}_loss_{}_overlap_{}'
        folder_name = folder_name.format((epoch + 1), loss, overlap_ratio)
        save_path = os.path.join(save_path, folder_name)
        
        # Create directory if neccessary.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Ensure directory is empty and save.
        if not os.listdir(save_path):
            torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder_model.pytorch'))
            torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder_model.pytorch'))
        else:
            print('Folder {} already exists and is not empty. Please specify a folder path with no files or subdirectories.'.format(save_path))
            

    # Inference.
    indices = np.random.permutation(dataset.dataset_size)[:batch_size]
    start_sequences = to_var(torch.from_numpy(dataset.x[indices]))
    predicted_batch = run(encoder, decoder, start_sequences, 20, init_hidden_once=True)
    
    if overlap_ratio > 0.0:
        first_magnitudes = predicted_batch[0]
        print(first_magnitudes.shape)
        first_magnitudes = first_magnitudes.reshape((-1, sequence_length, feature_size))
        first_magnitudes = overlap_and_add(first_magnitudes, dataset.offset)
        print(first_magnitudes.shape)
    
        
    # Reconstruct phase and generate samples.
    if griffin_lim_phase:
        predicted_audio = griffin_lim(first_magnitudes,
                                      n_iter=griffin_lim_iterations,
                                      window='hann',
                                      n_fft=fft_size,
                                      hop_length=hop_length)
    else:
        if lws_processor is None:
            lws_processor = lws.lws(fft_size,
                                    hop_length,
                                    mode=mode,
                                    perfectrec=perfect_reconstruction)
        predicted_stfts = lws_processor.run_lws(first_magnitudes)
        predicted_audio = lws_processor.istft(predicted_stfts)

    # Get file name and write to wav.
    librosa.output.write_wav(file_name, predicted_audio, dataset.sample_rate)
    print("Done.")
