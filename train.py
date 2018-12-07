import time
import sys

import torch
import torch.nn as nn
from torch import optim
from scipy.spatial import distance

from model import EncoderDecoder
import data

RHO = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_progress(start, count, iteration, loss):
    """Print progress as <Epoch, Iteration, Elapsed, Loss>"""
    elapsed = time.time() - start
    epoch = int(iteration / count)
    print("Epoch %d, Iteration %d, Elapsed: %ds, Loss: %.2f" % (epoch, iteration, elapsed, loss))


def training_loop(model, triples, iterations, print_every, adj_model):
    """Training loop, running for given amount of iterations on given triples"""
    triple_count = len(triples)
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)
    loss_function = nn.MSELoss()
    start = time.time()

    for iteration in range(iterations):
        x, z, y = triples[iteration % triple_count]
        loss = train(
            model,
            optimizer,
            loss_function,
            x,
            z,
            y,
        )

        if iteration % print_every == 0 or iteration % triple_count == 0:
            evaluate_gre(model, adj_model)
            print_progress(start, triple_count, iteration, loss)


def train(model, optimizer, loss_function, x, z, y):
    """Run a single training iteration"""
    model.zero_grad()

    y_pred = model(x, z)

    loss = loss_function(y_pred, y)
    loss.backward()

    optimizer.step()

    return loss


def compute_cosine(t1, t2):
    """Compute cosine similarity between two pytorch tensors"""
    return distance.cosine(t1.numpy(), t2.numpy())


def evaluate_gre(model, adj_model, gre=None):
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

        adj_ant_pred = model(adj.embedding, gate)

        closest_dist = sys.maxsize
        closest_word = ""
        for opt_str in options:
            opt = adj_model.adj_from_name(opt_str)
            # prediction needs detach since torch can do numpy() when var requires grad
            dist = compute_cosine(adj_ant_pred.detach(), opt.embedding)
            print(dist)
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
    start = time.time()
    triples, adj_model = data.build_triples_and_adj_model(restricted=False)
    print("Built input/adj_model in %ds" % (time.time() - start))
    print("Training on: ", device)
    for x, y, z in triples:
        x.to(device)
        y.to(device)
        z.to(device)


    model = EncoderDecoder()
    # make enc/dec uses doubles since our input is doubles
    model.double()
    model.to(device)

    training_loop(model, triples, 20000, 100, adj_model)
    evaluate_gre(model, adj_model)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
