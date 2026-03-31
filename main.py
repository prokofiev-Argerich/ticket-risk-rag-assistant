"""Main functionality for the ticket risk project."""

import os
import random
import logging
import argparse
import torch
import numpy as np

from src.data import load_data
from src.trainer import train
from src.metrics import evaluate
from src.model import MetaphorModel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("training")
    group.add_argument("--lr", type=float, default=3e-5)
    group.add_argument("--seed", type=int, default=1)
    group.add_argument("--metaphor_weight", type=float, default=0.8,
                       help="Temporary compatibility name. Used as positive risk-class weight.")
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--train_steps", type=int, default=1000)
    group.add_argument("--output", type=str, default="ticket_eval.tsv")
    group.add_argument("--alpha", type=float, default=0.5,
                       help="Weight of risk detection loss.")
    group.add_argument("--beta", type=float, default=0.5,
                       help="Weight of priority scoring loss.")
    args = vars(parser.parse_args())

    for key, value in args.items():
        logging.info(f"--{key}: {value}")

    set_seed(args["seed"])

    train_set, dev, test = load_data(args["batch_size"])

    model = MetaphorModel()
    if torch.cuda.is_available():
        model.cuda()

    best_model = train(model, train_set, dev, **args)

    logging.info("Best Model")
    evaluate(best_model, test, "test", args["output"])
    torch.save(best_model, "model.pt")
