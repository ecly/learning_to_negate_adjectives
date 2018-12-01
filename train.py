import time
import sys

import torch.nn as nn
from torch import optim
from scipy.spatial import distance

from model import Encoder, Decoder
import data

RHO = 0.95

def print_progress(start, count, iteration, loss):
    """Print progress as <Epoch, Iteration, Elapsed, Loss>"""
    elapsed = time.time() - start
    epoch = int(iteration / count)
    print("Epoch %d, Iteration %d, Elapsed: %d, Loss: %.2f" % (epoch, iteration, elapsed, loss))


def training_loop(encoder, decoder, triples, iterations, print_every, adj_model):
    """Training loop, running for given amount of iterations on given triples"""
    triple_count = len(triples)
    encoder_optimizer = optim.Adadelta(encoder.parameters(), rho=RHO)
    decoder_optimizer = optim.Adadelta(decoder.parameters(), rho=RHO)
    loss_function = nn.MSELoss()
    start = time.time()

    for iteration in range(iterations):
        x, z, y = triples[iteration % triple_count]
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

        if iteration % print_every == 0 or iteration % triple_count == 0:
            evaluate_gre(encoder, decoder, adj_model)
            print_progress(start, triple_count, iteration, loss)


def train(encoder, decoder, enc_optim, dec_optim, loss_function, x, z, y):
    """Run a single training iteration"""
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    decoder_output = predict_ant(encoder, decoder, x, z)

    loss = loss_function(decoder_output, y)
    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss

def predict_ant(encoder, decoder, x, z):
    """
    Predict antonym with given encoder and decoder
    for input x and its gate vector z
    """
    h = encoder(x, z)
    return decoder(h, z)


def compute_cosine(t1, t2):
    """Compute cosine similarity between two pytorch tensors"""
    return distance.cosine(t1.numpy(), t2.numpy())


def evaluate_gre(encoder, decoder, adj_model, gre=None):
    """
    Evaluate the given encoder, decoder according to GRE
    question set. The given adj_model is needed to compute
    embeddings for the GRE adjectives.

    Optionally takes a loaded GRE dataset to avoid loading
    multiple times, if evaluation is ran repeatedly.
    """
    gre_data = data.load_gre_test_set(adj_model) if gre is None else gre
    right = []
    wrong = []
    for test in gre_data:
        adj_str, options, ant = test
        adj = adj_model.adj_from_name(adj_str)
        gate = data.find_gate_vector(adj, adj_model)

        adj_ant_pred = predict_ant(encoder, decoder, adj.embedding, gate)

        closest_dist = sys.maxsize
        closest_word = ""
        for opt_str in options:
            opt = adj_model.adj_from_name(opt_str)
            # prediction needs detach since torch can do numpy() when var requires grad
            dist = compute_cosine(adj_ant_pred.detach(), opt.embedding)
            if dist < closest_dist:
                closest_dist = dist
                closest_word = opt_str

        if closest_word == ant:
            right.append(test)
        else:
            wrong.append(test)


    acc = len(right) / len(gre_data)
    print("GRE question accuracy: %.2f" % acc)


def main():
    triples, adj_model = data.build_triples_and_adj_model(restricted=False)
    encoder = Encoder()
    decoder = Decoder()
    # make enc/dec uses doubles since our input is doubles
    encoder.double()
    decoder.double()

    evaluate_gre(encoder, decoder, adj_model)
    training_loop(encoder, decoder, triples, 20000, 100, adj_model)
    evaluate_gre(encoder, decoder, adj_model)


if __name__ == "__main__":
    main()
