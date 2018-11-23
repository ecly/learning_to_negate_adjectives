import random
import time
import sys
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import gensim

from model import Encoder, Decoder
import data

RHO = 0.95

def training_loop(encoder, decoder, model, pairs, iterations):
    pair_count = len(pairs)
    encoder_optimizer = optim.Adadelta(encoder.parameters(), rho=RHO)
    decoder_optimizer = optim.Adadelta(decoder.parameters(), rho=RHO)
    loss_function = nn.MSELoss()

    for iteration in range(iterations):
        idx = iteration % pair_count
        x, z, y = pairs[idx]
        loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, x, z, y)

        if iteration % pair_count == 0:
            epoch = int(iteration/pair_count)
            print("Epoch %d, Iteration %d, Loss: %.2f" % ( epoch, iteration, loss))


def train(encoder, decoder, enc_optim, dec_optim, loss_function, x, z, y):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    h = encoder(x,z)
    decoder_output = decoder(h, z)

    loss = loss_function(decoder_output, y)
    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        data.GOOGLE_NEWS_PATH, binary=True
    )
    adj_dict = data.build_adjective_dict(model)
    filtered_words = data.load_gre_filtered_words()
    pairs = data.build_training_pairs(adj_dict, model, filtered_words)
    encoder = Encoder()
    decoder = Decoder()

    training_loop(encoder, decoder, model, pairs, 100)


if __name__ == "__main__":
    main()
