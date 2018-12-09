"""
Module for loading/saving/training a adjective negation model.
Also includes evaluation code.

Requires ~7GB of either CUDA or regular Memory depending
on the device used for training.
"""
import time
import sys
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from model import EncoderDecoder
import data

RHO = 0.95
BATCH_SIZE = 48
EPOCHS = 200
MODEL_PATH = "adjective_negation_model.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_progress(start, epoch, batch, loss):
    """Print progress as <Epoch, Batch, Elapsed, Loss>"""
    elapsed = time.time() - start
    print("Epoch %d, Batch %d, Elapsed: %ds, Loss: %.2f" % (epoch, batch, elapsed, loss))


def training_loop(model, optimizer, data_loader, epochs, adj_model):
    """Training loop, running for given amount of iterations on given triples"""
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)
    loss_function = nn.MSELoss()
    start = time.time()

    for epoch in range(epochs):
        print("Epoch %d:" % epoch)
        with torch.set_grad_enabled(False):
            evaluate_gre(model, adj_model)
        for batch_idx, batch in enumerate(data_loader):
            loss = train(
                model,
                optimizer,
                loss_function,
                batch
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
    """
    Compute cosine similarity between two pytorch tensors
    Returns a regular python float where 1.0 means identical
    tensors and 0.0 means orthoganal tensors.
    """
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)
    return F.cosine_similarity(tensor1, tensor2, dim=0).item()


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
        adj_str, options, answer = test
        adj = adj_model.adj_from_name(adj_str)
        gate = data.find_gate_vector(adj, adj_model)

        x, y = adj.embedding.to(device), gate.to(device)
        ant_pred = model(x, y)

        most_similar = 0
        most_similar_word = ""
        for opt_str in options:
            opt = adj_model.adj_from_name(opt_str)
            similarity = compute_cosine(ant_pred, opt.embedding)
            if similarity > most_similar:
                most_similar = similarity
                most_similar_word = opt_str

        if most_similar_word == answer:
            right.append(test)
        else:
            wrong.append(test)


    acc = len(right) / len(gre_data)
    print("GRE question accuracy: %.2f" % acc)


def main():
    """Build dataset and train model"""
    start = time.time()
    print("Building dataset and adjectives")
    dataset, adj_model = data.build_dataset_and_adj_model(restricted=False)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Built dataset and adjectives in %ds" % (time.time() - start))

    model = EncoderDecoder()
    model.double() # use doubles since our input is doubles
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)

    # if a model exists, load that one
    if os.path.isfile(MODEL_PATH):
        print("Loading model from", MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    try:
        print("Training on", device.type.upper())
        training_loop(model, optimizer, data_loader, EPOCHS, adj_model)
        with torch.set_grad_enabled(False):
            evaluate_gre(model, adj_model)
    finally:
        # Always save model
        print("Saving model to", MODEL_PATH)
        torch.save(
            {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict()}
            , MODEL_PATH)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
