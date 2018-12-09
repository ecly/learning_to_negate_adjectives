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
from torch.utils.data import DataLoader
from torch import optim

from model import EncoderDecoder
import evaluate
import data

RHO = 0.95
BATCH_SIZE = 48
EPOCHS = 200
MODEL_PATH = "adjective_negation_model.tar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_progress(start, epoch, batch, loss):
    """Print progress as <Epoch, Batch, Elapsed, Loss>"""
    elapsed = time.time() - start
    print(
        "Epoch %d, Batch %d, Elapsed: %ds, Loss: %.2f" % (epoch, batch, elapsed, loss)
    )


def training_loop(model, optimizer, data_loader, epochs=EPOCHS):
    """Training loop, running for given amount of iterations on given triples"""
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)
    loss_function = nn.MSELoss()
    start = time.time()

    for epoch in range(epochs):
        print("Epoch %d:" % epoch)
        for batch_idx, batch in enumerate(data_loader):
            loss = train(model, optimizer, loss_function, batch)
            print_progress(start, epoch, batch_idx, loss)


def train(model, optimizer, loss_function, batch):
    """Run a single training iteration"""
    x, z, y = batch
    x, z, y = x.to(DEVICE), z.to(DEVICE), y.to(DEVICE)

    model.zero_grad()
    y_pred = model(x, z)
    loss = loss_function(y_pred, y)
    loss.backward()

    optimizer.step()

    return loss


def initialize_model(model_path=None, device=DEVICE):
    """
    Initializes a model and an optimizer.
    If a model_path is given, state_dict for EncoderDecoder
    model as well as state_dict for optimizer are loaded in.

    If the a device is given, model will be moved to the given
    device. If device is 'cuda' and multiple devices are given,
    the model will be wrapped in DataParallel

    Returns (model, optimizer)
    """
    model = EncoderDecoder()
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)
    if model_path is not None:
        model = EncoderDecoder()
        print("Loading model from", model_path)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.double()  # use doubles since our input is doubles
    model.to(device)
    return model, optimizer


def main():
    """Build dataset and train model"""
    start = time.time()
    print("Building dataset and adjectives")
    dataset = data.build_dataset(restricted=False)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Built dataset and adjectives in %ds" % (time.time() - start))
    model, optimizer = initialize_model(MODEL_PATH)

    try:
        print("Training on", DEVICE.type.upper())
        training_loop(model, optimizer, data_loader)
    finally:
        # Always save model
        print("Saving model to", MODEL_PATH)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            MODEL_PATH,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        device_type = sys.argv[1].lower()
        assert device_type in ["cpu", "cuda"]
        DEVICE = torch.device(device_type)

    main()
