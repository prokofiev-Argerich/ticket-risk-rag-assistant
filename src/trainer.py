"""Training and evaluation functions for the ticket risk project."""

import gc
import copy
import logging
import numpy as np
import torch

from torch.nn import MSELoss, BCELoss
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from src.metrics import evaluate


def train(model, train, dev, train_steps, lr, metaphor_weight, alpha, beta,
          **kwargs):
    """
    Train model on risk-style dual-task data.

    Args:
        model (nn.Module): initialised model, untrained
        train (DataLoader): training data
        dev (DataLoader): validation data
        train_steps (int): number of updates to train for
        lr (float): learning rate
        metaphor_weight (float): compatibility name; positive risk-class weight
        alpha (float): weight for the risk detection task
        beta (float): weight for the priority scoring task

    Returns:
        model: current model after training
    """
    optimiser = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=int(train_steps * .1), num_training_steps=train_steps
    )
    train_iter = iter(train)
    loss_fn_priority = MSELoss()
    risk_losses, priority_losses = [], []

    for x in range(train_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train)
            risk_losses = []
            priority_losses = []
            batch = next(train_iter)

        model.train()
        optimiser.zero_grad()
        risk_output, priority_output = model(batch.tokens, batch.mask)

        risk_labels = batch.bert_metaphoricity_labels.view(-1)
        risk_output = risk_output.cpu().contiguous().view(-1)

        valid_mask = risk_labels != -2
        risk_labels = risk_labels[valid_mask]
        risk_output = risk_output[valid_mask]

        weights = torch.where(
            risk_labels == 1,
            torch.full_like(risk_labels, metaphor_weight),
            torch.full_like(risk_labels, 1 - metaphor_weight)
        )

        loss_fn_risk = BCELoss(weight=weights)
        risk_loss = loss_fn_risk(risk_output, risk_labels)

        risk_losses.append(risk_loss.item())

        priority_labels = batch.bert_novelty_labels.view(-1)
        priority_output = priority_output.cpu().contiguous().view(-1)
        weights = copy.deepcopy(priority_labels)
        weights[weights != -2] = 1
        weights[weights == -2] = 0
        priority_loss = loss_fn_priority(priority_output * weights, priority_labels * weights)
        priority_losses.append(priority_loss.item())

        loss = alpha * risk_loss + beta * priority_loss
        loss.backward()
        optimiser.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if (x + 1) % 500 == 0:
            logging.info(
                f"Training, Risk Loss: {np.mean(risk_losses):.3f} "
                f"Priority Loss: {np.mean(priority_losses):.3f}"
            )
            evaluate(model, dev)
            torch.cuda.empty_cache()
    return model


def clean_object_from_memory(obj):
    """Clean Pytorch object from memory."""
    del obj
    gc.collect()
    torch.cuda.empty_cache()
