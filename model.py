import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch

CENTROID_BASIS_COUNT = 10
HIDDEN_SIZE = 600
EMBEDDING_SIZE = 300

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, input_size=EMBEDDING_SIZE):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.activation = nn.Sigmoid()
        self.encoder_weights = nn.Bilinear(input_size, input_size, hidden_size)

    def forward(self, x, z):
        return self.activation(self.encoder_weights(z, x))


class Decoder(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, output_size=EMBEDDING_SIZE):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.activation = nn.Sigmoid()
        self.decoder_weights = nn.Bilinear(hidden_size, hidden_size, output_size)

    def forward(self, h, z):
        return self.activation(self.encoder_weights(h, z))
