"""
Module for loading/saving/training a adjective negation model.
Also includes evaluation code.
"""
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from scipy.spatial import distance

from model import EncoderDecoder
import data

RHO = 0.95
BATCH_SIZE = 48
EPOCHS = 200
MODEL_PATH = "adjective_negation_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_progress(start, epoch, batch, loss):
    """Print progress as <Epoch, Batch, Elapsed, Loss>"""
    elapsed = time.time() - start
    print("Epoch %d, Batch %d, Elapsed: %ds, Loss: %.2f" % (epoch, batch, elapsed, loss))


def training_loop(model, data_loader, epochs, adj_model):
    """Training loop, running for given amount of iterations on given triples"""
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)
    loss_function = nn.MSELoss()
    start = time.time()

    for epoch in range(epochs):
        with torch.set_grad_enabled(False):
            evaluate_gre(model, adj_model)
        for batch_idx, batch in enumerate(data_loader):
            loss = train(
                model,
                optimizer,
                loss_function,
                batch.to(device)
            )
            print_progress(start, epoch, batch_idx, loss)


def train(model, optimizer, loss_function, batch):
    """Run a single training iteration"""
    x, z, y = batch
    x, z, y = x.to(device), z.to(device), y.to(device)

    model.zero_grad()
    y_pred = model(x, z)
    loss = loss_function(y_pred, y)
    loss.backward()

    optimizer.step()

    return loss


def compute_cosine(tensor1, tensor2):
    """Compute cosine similarity between two pytorch tensors"""
    return distance.cosine(tensor1.numpy(), tensor2.numpy())


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

        x, y = adj.embedding.to(device), gate.to(device)
        adj_ant_pred = model(x, y)

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
    """Build dataset and train model"""
    start = time.time()
    dataset, adj_model = data.build_dataset_and_adj_model(restricted=False)

    print("Built input/adj_model in %ds" % (time.time() - start))
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = EncoderDecoder()
    # model.load_state_dict(torch.load(MODEL_PATH)
    # make model use doubles since our input is doubles
    model.double()
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    print("Training on: ", device)
    training_loop(model, data_loader, EPOCHS, adj_model)
    with torch.set_grad_enabled(False):
        evaluate_gre(model, adj_model)
    torch.save(model.state_dict(), MODEL_PATH)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
