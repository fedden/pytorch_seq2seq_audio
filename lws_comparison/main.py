from __future__ import print_function

import torch
import torch.nn as nn
import lws
import librosa
import argparse
import numpy as np
from neural_methods import train, run
from utils import get_file_name
from audio import AudioDataset, griffin_lim
from model import EncoderRNN, AttentionDecoderRNN, to_var

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fft_size', required=True)
parser.add_argument('-l', '--lws_mags', required=True)
parser.add_argument('-g', '--griffin_lim_phase', required=True)
parser.add_argument('-i', '--griffin_lim_iterations', required=True)
parser.add_argument('-p', '--perfect_reconstruction', required=True)
parser.add_argument('-m', '--mode', required=True)
parser.add_argument('-c', '--cuda_device', required=True)
args = vars(parser.parse_args())

fft_size = int(args['fft_size'])
lws_mags = bool(args['fft_size'])
griffin_lim_phase = bool(args['griffin_lim_phase'])
griffin_lim_iterations = int(args['griffin_lim_iterations'])
perfect_reconstruction = bool(args['perfect_reconstruction'])
mode = args['mode']
cuda_device = int(args['cuda_device'])
sequence_length = 150
batch_size = 8
number_epochs = 20
feature_size = input_size = hidden_size = (fft_size // 2) + 1
hop_length = fft_size // 4
path = "./massive_chops/trimmed/vocals_trimmed.wav"

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
                           path=path)

    # Train.
    previous_epoch = -1
    epoch_str = "epoch {}/{}, loss {}"
    for x, y, epoch in dataset.get_next_batch(number_epochs):

        loss = train(encoder, decoder, encoder_optimiser, decoder_optimiser,
                     x, y, criterion, 5.0)
        print(epoch_str.format(epoch, number_epochs, loss), end="\r")

        if epoch != previous_epoch:
            previous_epoch = epoch
            print('')

    # Inference.
    indices = np.random.permutation(dataset.dataset_size)[:batch_size]
    start_sequences = to_var(torch.from_numpy(dataset.x[indices]))
    predicted_batch = run(encoder, decoder, start_sequences, 20)
    first_magnitudes = predicted_batch[0]

    # Reconstruct phase and generate samples.
    if griffin_lim_phase:
        predicted_audio = griffin_lim(first_magnitudes,
                                      n_iter=griffin_lim_iterations,
                                      window='hann',
                                      n_fft=fft_size,
                                      hop_length=hop_length)
    else:
        predicted_stfts = lws_processor.run_lws(first_magnitudes)
        predicted_audio = lws_processor.istft(predicted_stfts)

    # Get file name and write to wav.
    file_name = get_file_name(fft_size, lws_mags, griffin_lim_phase,
                              griffin_lim_iterations, perfect_reconstruction,
                              mode)
    librosa.output.write_wav(file_name, predicted_audio, dataset.sample_rate)
    print("Done.")
