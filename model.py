import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch

HIDDEN_SIZE = 600
EMBEDDING_SIZE = 300

class EncoderDecoder(nn.Module):
    """
    Combination of Encoder and Decoder.
    Produces a predicted antonym y, given and
    adjective x and a gate vector z.
    """
    def __init__(self, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hidden_size, embedding_size)
        self.decoder = Decoder(hidden_size, embedding_size)

    def forward(self, x, z):
        h = self.encoder(x, z)
        return self.decoder(h, z)


class Encoder(nn.Module):
    """Encoder producing hidden state h given adjective x and gate z"""
    def __init__(self, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.activation = nn.Sigmoid()
        self.encoder_weights = nn.Bilinear(embedding_size, embedding_size, hidden_size)

    def forward(self, x, z):
        return self.activation(self.encoder_weights(z, x))


class Decoder(nn.Module):
    """Deocder predicting antonym y given adjective x and hidden state h"""
    def __init__(self, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.activation = nn.Sigmoid()
        self.decoder_weights = nn.Bilinear(hidden_size, embedding_size, embedding_size)

    def forward(self, h, z):
        return self.activation(self.decoder_weights(h, z))
