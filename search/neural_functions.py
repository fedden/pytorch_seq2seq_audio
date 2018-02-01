import torch
from torch.autograd import Variable
import numpy as np
from models import EncoderRNN, AttentionDecoderRNN
from dataset import AudioDataset
from utils import to_var, griffin_lim


def train(encoder,
          decoder,
          encoder_optimiser,
          decoder_optimiser,
          input_batch,
          target_batch,
          criterion,
          clip_threshold):

    # Zero the gradients of both optimisers.
    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    batch_size, sequence_length, fft_size = tuple(input_batch.size())

    # Run the input batch of fft sequences through the encoder.
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_batch, encoder_hidden)

    # Prepare the input and output variables.
    decoder_input = input_batch
    decoder_hidden = encoder_hidden[:decoder.number_layers]

    decoder_output, decoder_hidden, decoder_attention = \
        decoder(decoder_input, decoder_hidden, encoder_outputs)

    # How close were we to the targets?
    loss = criterion(decoder_output, target_batch)
    loss.backward()

    # Clip the gradient norms.
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip_threshold)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip_threshold)

    # Update parameters with optimisers.
    encoder_optimiser.step()
    decoder_optimiser.step()

    return loss.data[0]


def run(encoder,
        decoder,
        start_sequences,
        amount_frames):

    encoder.train(False)
    decoder.train(False)

    # Get random sequence from dataset object.
    batch_size, sequence_length, fft_size = start_sequences.size()
    model_input = start_sequences.view(batch_size, sequence_length, fft_size)

    frames = np.zeros((amount_frames, batch_size, sequence_length, fft_size))

    # Repeatedly sample the RNN and get the output.
    for i in range(amount_frames):

        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(model_input, encoder_hidden)

        decoder_input = model_input
        decoder_hidden = encoder_hidden[:decoder.number_layers]

        decoder_output, decoder_hidden, _ = \
            decoder(decoder_input, decoder_hidden, encoder_outputs)

        model_input = decoder_output
        frames[i] = Variable(decoder_output.data).cpu().data.numpy()

    encoder.train(True)
    decoder.train(True)

    return frames.reshape((batch_size, -1, fft_size))


def experiment(attention_method,
               fft_size,
               number_layers,
               dropout,
               sequence_length,
               clip_threshold,
               criterion,
               learning_rate,
               decoder_learning_ratio,
               batch_size,
               number_epochs,
               url,
               path):

    seed = 42
    feature_size = hidden_size = (fft_size // 2) + 1
    hop_length = fft_size // 4

    # Construct RNNs and optimisers.
    encoder = EncoderRNN(input_size=feature_size,
                         batch_size=batch_size,
                         hidden_size=feature_size,
                         number_layers=number_layers,
                         dropout=dropout)
    decoder = AttentionDecoderRNN(attention_method,
                                  batch_size=batch_size,
                                  hidden_size=hidden_size,
                                  output_size=feature_size,
                                  number_layers=number_layers,
                                  dropout=dropout)

    # Enable GPU tensors provided GPUs actually exist!
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    np.random.seed(seed)

    encoder_optimiser = torch.optim.Adam(encoder.parameters(),
                                         lr=learning_rate)
    decoder_learning_rate = learning_rate * decoder_learning_ratio
    decoder_optimiser = torch.optim.Adam(decoder.parameters(),
                                         lr=decoder_learning_rate)

    data = AudioDataset(fft_size=fft_size,
                        hop_length=hop_length,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        url=url,
                        path=path,
                        limit=20)

    indices = np.random.permutation(data.dataset_size)[:batch_size]
    start_sequences = to_var(torch.from_numpy(data.x[indices]))

    get_batch = data.get_next_batch(number_epochs)

    for x, y, epoch in get_batch:
        train(encoder,
              decoder,
              encoder_optimiser,
              decoder_optimiser,
              x,
              y,
              criterion,
              clip_threshold)

    amount_frames = 100
    predicted_batch = run(encoder,
                          decoder,
                          start_sequences,
                          amount_frames)
    predicted_audio = []
    for i in range(len(predicted_batch)):
        predicted_audio.append(griffin_lim(predicted_batch[i].T,
                                           n_iter=100,
                                           window='hann',
                                           n_fft=fft_size,
                                           hop_length=hop_length))
    return np.array(predicted_audio)
