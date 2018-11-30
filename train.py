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


def training_loop(encoder, decoder, model, triples, iterations):
    triple_count = len(triples)
    encoder_optimizer = optim.Adadelta(encoder.parameters(), rho=RHO)
    decoder_optimizer = optim.Adadelta(decoder.parameters(), rho=RHO)
    loss_function = nn.MSELoss()

    for iteration in range(iterations):
        idx = iteration % triple_count
        x, z, y = triples[idx]
        loss = train(
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            loss_function,
            x,
            z,
            y,
        )

        if iteration % triple_count == 0:
            epoch = int(iteration / triple_count)
            print("Epoch %d, Iteration %d, Loss: %.2f" % (epoch, iteration, loss))


def train(encoder, decoder, enc_optim, dec_optim, loss_function, x, z, y):
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    h = encoder(x, z)
    decoder_output = decoder(h, z)

    loss = loss_function(decoder_output, y)
    print("%.2f" % loss)
    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss


def evaluate_gre(adj_model, encoder, decoder):
    gre = data.load_gre_test_set()
    right = []
    wrong = []
    for test in gre:
        adj, options, ant = test

    acc = len(right) / len(gre)
    print("GRE question accuracy: %.2f" % acc)


def main():
    triples, adj_model = data.build_triples_and_adj_model(restricted=False)
    encoder = Encoder()
    decoder = Decoder()
    encoder.double()
    decoder.double()

    training_loop(encoder, decoder, adj_model, triples, 100)
    # evaluate_gre(adj_model, encoder, decoder)


if __name__ == "__main__":
    main()
