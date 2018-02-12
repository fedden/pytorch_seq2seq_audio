# NumPy on steriods.
import torch
import torch.nn as nn
import torch.nn.functional as F

from search.utils import to_var


class EncoderRNN(nn.Module):

    """The encoder takes a batch of FFT frame sequences and returns encodings.

    There are no embeddings as the inputs are already real values, and the
    objective is to take FFT frames sequences (batch_size, sequence_length,
    fft_size) and return the encodings (batch_size, sequence_length,
    hidden_size).
    """

    def __init__(self,
                 input_size,
                 batch_size,
                 hidden_size,
                 number_layers=1,
                 dropout=0.1,
                 bi_directional=True,
                 rnn=nn.GRU):
        """The __init__ function.

        Sets up the encoder with the specified amount of layers and hidden
        size. Uses a different mask for each timestep, which has been shown to
        perform worse than if the mask was the same, for diffent layers at
        different timesteps as in https://arxiv.org/abs/1512.05287. So be wary
        of this! To entirely cancel out the dropout, just set dropout to zero.

        Arguments:
          input_size: Size of the input FFT frames.
          hidden_size: Size of the hidden learnable weights for the RNN.
          number_layers: How many layers the encoding RNN should have.
          dropout: Zero for no dropout, one for complete dropout.
          bi_directional: Whether or not the RNN should run in two directions.
          rnn_type: Pass a RNN module here such as nn.GRU or nn.LSTM.
        """

        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.dropout = dropout
        self.bi_directional = bi_directional
        self.rnn = rnn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=number_layers,
                       bidirectional=bi_directional,
                       batch_first=True)

    def forward(self, x, hidden):
        """The forward propoagation function.

        The
        """

        x, hidden = self.rnn(x, hidden)

        # linearly sum bi-directional outputs if appropriate.
        if self.bi_directional:
            forward = x[:, :, :self.hidden_size]
            backward = x[:, :, self.hidden_size:]
            x = forward + backward

        return x, hidden

    def init_hidden(self):
        """"""
        amount = 2 if self.bi_directional else 1
        tensor = torch.FloatTensor(self.number_layers * amount,
                                   self.batch_size,
                                   self.hidden_size)
        return to_var(tensor.fill_(0))


class Attention(nn.Module):

    """An attention module that can switch between different scoring methods.

    The general form of the attention calculation uses the target decoder
    hidden state and the encoder state.

    Find the scoring methods here: https://arxiv.org/abs/1508.04025. The
    options are; 'dot', a dot product between the decoder and encoder state;
    'general', the dot product between the decoder state and a linear transform
    of the encoder state; 'concat', a dot product between a new parameter 'v'
    and a linear transform of the states concatenated together. Finally the
    result is normalised.
    """

    def __init__(self, method, batch_size, hidden_size, bias=True):
        super(Attention, self).__init__()

        self.method = method
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attention = nn.Linear(hidden_size, hidden_size, bias=bias)

        elif self.method == 'concat':
            self.attention = nn.Linear(hidden_size * 2, hidden_size, bias=bias)
            self.v = nn.Parameter(
                torch.FloatTensor(batch_size, 1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        if self.method == 'general':
            attention_energies = self.attention(
                encoder_outputs).transpose(2, 1)
            attention_energies = hidden.bmm(attention_energies)

        elif self.method == 'concat':
            # Broadcast hidden to encoder_outputs size.
            shape = encoder_outputs.data.new(encoder_outputs.size()).fill_(1)
            hidden *= to_var(shape)
            concat = torch.cat((hidden, encoder_outputs), -1)
            attention_energies = self.attention(concat)

            # Swap the second and first dimensions.
            attention_energies = attention_energies.transpose(2, 1)
            attention_energies = self.v.bmm(attention_energies)

        else:
            # Method is 'dot'.
            encoder_outputs = encoder_outputs.transpose(2, 1)
            attention_energies = hidden.bmm(encoder_outputs)

        return F.softmax(attention_energies)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)

        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.dot(encoder_output)

        elif self.method == 'concat':
            concat = torch.cat((hidden, encoder_output), 1)
            energy = self.attention(concat)
            energy = self.v.dot(energy)

        return energy


class AttentionDecoderRNN(nn.Module):

    """Luong Attention Decoder RNN module.

    This decoder plugs in the Attention module ater the RNN to calculate the
    """

    def __init__(self,
                 attention_method,
                 batch_size,
                 hidden_size,
                 output_size,
                 number_layers=1,
                 dropout=0.1):

        super(AttentionDecoderRNN, self).__init__()

        self.attention_method = attention_method
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.dropout = dropout

        self.rnn = nn.GRU(hidden_size,
                          hidden_size,
                          number_layers,
                          dropout=dropout,
                          batch_first=True)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        self.attention = Attention(attention_method, batch_size, hidden_size)

    def forward(self, input_sequence, last_hidden, encoder_outputs):

        # Get current hidden state from input sequence and last hidden state.
        rnn_output, hidden = self.rnn(input_sequence, last_hidden)

        # Calculate attention from the current RNN state and all encoder
        # outputs.
        attention_weights = self.attention(rnn_output, encoder_outputs)

        # bmm = batch matrix matrix product. Apply attention to encoder outputs
        # to get the weighted average.
        context = attention_weights.bmm(encoder_outputs)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5).
        concat_input = torch.cat((rnn_output, context), -1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next frame (Luong eq. 6, without softmax).
        output = self.output(concat_output)

        # Return final output, hidden state, and attention weights. We can use
        # the attention weights for visualisation.
        return output, hidden, attention_weights
