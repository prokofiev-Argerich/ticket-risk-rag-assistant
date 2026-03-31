"""Prediction functionality for the ticket risk project."""

import jieba
import os
import logging
import argparse
import csv
import torch

from torch.utils.data import DataLoader, SequentialSampler
from src.data import CustomDataset
from src.metrics import evaluate

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("model")
    group.add_argument("--model_name", type=str, default="model.pt")
    group.add_argument("--output", type=str, default="ticket_predictions.tsv")
    args = vars(parser.parse_args())
    logging.info(args)

    model = torch.load(args["model_name"], map_location="cpu", weights_only=False)

    snts, risk_labels, priority_labels = [], [], []
    with open("data/ticket_samples.csv", encoding="utf-8") as csvfile:
        corpus = csv.reader(csvfile, delimiter=',')
        next(corpus)
        for row in corpus:
            if not row:
                continue
            sentence = jieba.lcut(row[0].strip())
            sentence = [x for x in sentence if x.strip()]
            snts.append(sentence)
            risk_labels.append([0 for _ in sentence])
            priority_labels.append([0 for _ in sentence])

    dataset = CustomDataset(list(snts), list(risk_labels), list(priority_labels))

    sampler = SequentialSampler(dataset)
    ticket_set = DataLoader(
        dataset, batch_size=32, sampler=sampler,
        collate_fn=CustomDataset.collate_fn
    )

    evaluate(model, ticket_set, output_filename=args["output"], no_labels=True)
