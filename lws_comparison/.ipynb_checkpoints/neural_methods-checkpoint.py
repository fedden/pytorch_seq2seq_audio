import os
import torch
import torch.nn as nn
import numpy as np
from logger import TrainLogger
from torch.autograd import Variable
from model import EncoderRNN, AttentionDecoderRNN


def load_model(settings, encoder, decoder):
    encoder_path = os.path.join(settings.load_path, 'encoder_model.pytorch')
    decoder_path = os.path.join(settings.load_path, 'decoder_model.pytorch')
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder


def save_model(settings, encoder, decoder):

    # Ensure directory is empty and save.
    if not os.listdir(settings.save_path):
        torch.save(encoder.state_dict(), os.path.join(settings.save_path, 'encoder_model.pytorch'))
        torch.save(decoder.state_dict(), os.path.join(settings.save_path, 'decoder_model.pytorch'))
    else:
        save_str =  'Folder {} already exists and is not empty. '
        save_str += 'Please specify a folder path with no files or subdirectories.'
        print(save_str.format(settings.save_path))


def get_model_and_optimisers(settings):
    
    # Construct RNNs and optimisers.
    encoder = EncoderRNN(input_size=settings.feature_size,
                         batch_size=settings.batch_size,
                         hidden_size=settings.feature_size,
                         number_layers=settings.encoder_layers,
                         dropout=settings.dropout)
    decoder = AttentionDecoderRNN('general',
                                  batch_size=settings.batch_size,
                                  hidden_size=settings.hidden_size,
                                  output_size=settings.feature_size,
                                  number_layers=settings.decoder_layers,
                                  dropout=settings.dropout)
    
    # Use L2 loss to measure performance.
    criterion = nn.MSELoss()

    # Enable GPU tensors provided GPUs actually exist!
    # Instanciate the models on the GPU.
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Both the encoder have different optimsisers (and different learning rates)
    encoder_optimiser = torch.optim.Adam(encoder.parameters(), 
                                         lr=settings.learning_rate)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), 
                                         lr=settings.learning_rate * 5)
    
    return encoder, decoder, criterion, encoder_optimiser, decoder_optimiser


def train_batch(encoder,
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


def train(encoder, 
          decoder, 
          encoder_optimiser, 
          decoder_optimiser,
          dataset,
          settings,
          criterion):
    
    log = TrainLogger(len(dataset), settings.epoch, settings.number_epochs)
    
    for x, y in dataset.get_next_batch(settings.number_epochs):
        
        log.start_iteration()
        loss = train_batch(encoder, decoder,
                           encoder_optimiser, 
                           decoder_optimiser, 
                           x, y, criterion, 5.0)
        log.end_iteration(loss)
    
    
def inference(encoder,
              decoder,
              start_sequences,
              amount_frames,
              init_hidden_once=True):
    encoder.train(False)
    decoder.train(False)

    # Get random sequence from dataset object.
    batch_size, sequence_length, fft_size = start_sequences.size()
    model_input = start_sequences.view(batch_size, sequence_length, fft_size)

    frames = np.zeros((amount_frames, batch_size, sequence_length, fft_size))
    
    if init_hidden_once:
        encoder_hidden = encoder.init_hidden()

    # Repeatedly sample the RNN and get the output.
    for i in range(amount_frames):

        if not init_hidden_once:
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
