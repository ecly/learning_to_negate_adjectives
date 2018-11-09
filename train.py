import random
import time
import sys
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import Encoder, Decoder

RHO = 0.95

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


def training_loop(encoder, decoder, model, pairs, iterations):
    pair_count = len(pairs)
    encoder_optimizer = optim.Adadelta(encoder.parameters(), rho=RHO)
    decoder_optimizer = optim.Adadelta(decoder.parameters(), rho=RHO)
    loss_function = nn.MSELoss()

    for iteration in range(iterations):
        idx = iteration % pair_count
        x, y = pairs[idx]
        z = find_gate_vector(x, model)
        loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer, _loss_function, x, y, z)

        if iteration % pair_count == 0:
            epoch = int(iteration/pair_count)
            print("Epoch %d, Iteration %d, Loss: %.2f" % ( epoch, iteration, loss))


def train(encoder, decoder, enc_optim, dec_optim, loss_function, x, y, z):


def main():
    input_file = sys.argv[1]
    encoder = Encoder()
    encoder = Decoder()


if __name__ == "__main__":
    main()
