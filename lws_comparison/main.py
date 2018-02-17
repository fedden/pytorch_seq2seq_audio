from __future__ import print_function
import librosa
import torch
import torch.nn as nn
from neural_methods import train, run
from utils import get_file_name
from audio import AudioDataset, griffin_lim
from model import EncoderRNN, AttentionDecoderRNN, to_var
import numpy as np
import lws

# Hyper parameters.
fft_size = 1024
lws_mags = True
griffin_lim_phase = True
griffin_lim_iterations = 100
perfect_reconstruction = True
mode = 'music'

sequence_length = 150
criterion = nn.MSELoss()
batch_size = 8
number_epochs = 20

feature_size = input_size = hidden_size = (fft_size // 2) + 1
hop_length = fft_size // 4
path = "./massive_chops/trimmed/vocals_trimmed.wav"

if lws_mags:
    lws_processor = lws.lws(fft_size,
                            hop_length,
                            mode='music',
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

# Enable GPU tensors provided GPUs actually exist!
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    criterion.cuda()
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=0.00001)
decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=0.00001 * 5.0)

dataset = AudioDataset(fft_size=fft_size,
                       hop_length=hop_length,
                       batch_size=batch_size,
                       lws_processor=lws_processor,
                       sequence_length=sequence_length,
                       path=path)

# Train.
previous_epoch = -1
for x, y, epoch in dataset.get_next_batch(number_epochs):

    loss = train(encoder, decoder, encoder_optimiser, decoder_optimiser,
                 x, y, criterion, 5.0)
    print("epoch {}/{}, loss {}".format(epoch, number_epochs, loss), end="\r")

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
                          griffin_lim_iterations, perfect_reconstruction, mode)
librosa.output.write_wav(file_name, predicted_audio, dataset.sample_rate)
print("Done.")
