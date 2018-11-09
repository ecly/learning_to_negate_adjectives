import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch

CENTROID_BASIS_COUNT = 10
HIDDEN_SIZE = 600
EMBEDDING_SIZE = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_centroid(matrix):
    """
    Calculate centroid of numpy matrix

    Returns: 1D torch tensor.
    """

    return torch.mean(torch.from_numpy(matrix), dim=0)


def find_gate_vector(adjective, model):
    """
    TESTME
    """
    hyponym_count = len(adjective.hyponyms)
    hyponyms = list(adjective.hyponyms)

    if hyponym_count < CENTROID_BASIS_COUNT:
        neighbors = model.most_similar(topn=hyponym_count + CENTROID_BASIS_COUNT)
        relevant = list(filter(neighbors, lambda x: x not in adjective.hypoyms))
        missing_hyponyms = hyponym_count-CENTROID_BASIS_COUNT
        hyponyms = hyponyms + relevant[:missing_hyponyms]

    embeddings = list(map(hyponyms, model.get_vector))
    return calc_centroid(embeddings)


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
