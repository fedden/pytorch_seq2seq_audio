import torch
import numpy as np
from torch.autograd import Variable


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
