"""Evaluation functions for the ticket risk project."""

import logging
import scipy.stats
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error


def evaluate(model, dataloader, dataset_name="validation", output_filename=None,
             no_labels=False):
    """
    Evaluate model on validation/test data.

    Args:
        model (nn.Module): model to evaluate
        dataloader (DataLoader): contains the validation batches
        dataset_name: validation / test
        output_filename (string): filename to write predictions to
    """
    model.eval()

    pairs, trace = [], []
    pairs_priority = []

    for batch in dataloader:
        inputs = batch.tokens
        risk_outputs, priority_outputs = model(inputs, batch.mask)

        for i, (prediction_r, prediction_p, length) in enumerate(
            zip(risk_outputs, priority_outputs, batch.lengths)):
            prediction_r = torch.round(prediction_r[:length - 1].cpu().squeeze(-1)).tolist()
            prediction_r = transform(prediction_r, batch.mapping[i])
            target_r = batch.metaphoricity_labels[i]
            pairs.extend([(t, p) for t, p in zip(target_r, prediction_r) if t != -2])

            prediction_p = prediction_p[:length - 1].cpu().tolist()
            prediction_p = transform(prediction_p, batch.mapping[i])
            target_p = batch.novelty_labels[i]
            pairs_priority.extend([(t, p) for t, p in zip(target_p, prediction_p) if t != -2])

            trace.append((
                batch.sentences[i], target_r, prediction_r, target_p, prediction_p)
            )

    str_ = 'Validating' if dataset_name == 'validation' else 'Testing'

    if not no_labels:
        tgt, prd = zip(*pairs)
        p, r, f1, _ = precision_recall_fscore_support(tgt, prd, average="binary")
        logging.info(f"{str_}, Risk Detection, P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")

        tgt, prd = zip(*pairs_priority)
        priority_corr = scipy.stats.pearsonr(tgt, prd)[0]
        mae = mean_absolute_error(tgt, prd)
        logging.info(f"{str_}, Priority Score: Pearson's r {priority_corr}, MAE: {mae}")
    else:
        f1, priority_corr = 0.0, 0.0

    if output_filename is not None:
        with open(output_filename, 'w', encoding="utf-8") as f:
            if no_labels:
                f.write("sentences\tprediction_risk\tprediction_priority\n")
            else:
                f.write("sentences\ttarget_risk\tprediction_risk\ttarget_priority\tprediction_priority\n")
            for sentence, target_r, prediction_r, target_p, prediction_p in trace:
                t_r = " ".join([str(x) for x in target_r])
                p_r = " ".join([str(x) for x in prediction_r])
                t_p = " ".join([str(round(x, 3)) for x in target_p])
                p_p = " ".join([str(round(x, 3)) for x in prediction_p])
                if no_labels:
                    f.write(f"{' '.join(sentence)}\t{p_r}\t{p_p}\n")
                else:
                    f.write(f"{' '.join(sentence)}\t{t_r}\t{p_r}\t{t_p}\t{p_p}\n")
    return f1, priority_corr


def transform(prediction_sent, ids_sent):
    """Map subword predictions back to the original words."""
    mapping = {i: None for i in range(max(ids_sent) + 1)}
    for prediction_word, index in zip(prediction_sent, ids_sent):
        if mapping[index] is None:
            mapping[index] = prediction_word
    return [mapping[i] for i in range(max(ids_sent) + 1)]
