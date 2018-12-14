"""
Module for loading/saving/training an adjective negation model.

Requires ~7GB of either CUDA or regular Memory depending
on the device used for training.

Allows options (output, model, restricted, unsupervised, device).

Examples:
    python train.py --output adjective_negation_model_restricted.tar --restricted
    python train.py --model adjective_negation_model_unsupervised.tar --unsupervised
    python train.py --device cpu
"""
import time
import sys
import os.path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from model import EncoderDecoder
import data

RHO = 0.95
BATCH_SIZE = 48
EPOCHS = 200
MODEL_DEFAULT_PATH = "adjective_negation_model.tar"
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
    If a model_path is given, state_dicts for the EncoderDecoder
    model for the optimizer are loaded in.

    If the a device is given, model will be moved to the given
    device. Model will always be wrapped in `nn.DataParallel` for
    consistency when loading models across devices. This can however
    be a slowdown when running in single device environments.

    Returns (model, optimizer)
    """
    model = nn.DataParallel(EncoderDecoder())
    optimizer = optim.Adadelta(model.parameters(), rho=RHO)

    if model_path is not None and os.path.isfile(model_path):
        print("Loading model from", model_path)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # use doubles since our input tensors use doubles
    model.double()
    model.to(device)
    return model, optimizer


def prepare_arg_parser():
    """Create arg parser handling input/output and training conditions"""
    arg_parser = argparse.ArgumentParser(
        description="Trains an existing or new adjective negation model. "
        + "If no args are given, the model at %s will be used " % MODEL_DEFAULT_PATH
        + "if it exists or it will be created if it does not exist. "
        + "Defaults to standard training conditions."
        + "Defaults to standard training conditions."
    )
    arg_parser.add_argument(
        "-r",
        "--restricted",
        action="store_true",
        help="train under the 'restricted' condition",
    )
    arg_parser.add_argument(
        "-u",
        "--unsupervised",
        action="store_true",
        help="train under the 'unsupervised' condition",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        nargs=1,
        metavar="output-path",
        default=[MODEL_DEFAULT_PATH],
        help="save the model to the given path on exit",
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        nargs=1,
        metavar="model-path",
        default=[MODEL_DEFAULT_PATH],
        help="loads the given model for training if it exists",
    )
    arg_parser.add_argument(
        "-d",
        "--device",
        nargs=1,
        metavar="device",
        default=["cuda" if torch.cuda.is_available() else "cpu"],
        help="use the given device (cuda/cpu) for training",
    )
    return arg_parser


def main():
    """Build dataset according to args and train model"""
    args = prepare_arg_parser().parse_args()
    device = torch.device(args.device[0])
    restricted = args.restricted
    unsupervised = args.unsupervised
    model_path = args.model[0]
    output_path = args.output[0]

    start = time.time()
    print("Building dataset and adjectives")
    dataset = data.build_dataset(restricted=restricted, unsupervised=unsupervised)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Built dataset and adjectives in %ds" % (time.time() - start))
    model, optimizer = initialize_model(model_path, device)

    try:
        print("Training on", device.type.upper())
        training_loop(model, optimizer, data_loader)
    finally:
        # Always save model. This catches SIGINT kill signal.
        # If stopping a model running in the background use:
        # kill -s SIGINT <pid>
        print("Saving model to", model_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_path,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
