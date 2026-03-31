import torch
from src.data import CustomDataset
from torch.utils.data import DataLoader, SequentialSampler


def predict_tokens(tokens, model_path="model.pt"):
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    dataset = CustomDataset([tokens], [[0] * len(tokens)], [[0] * len(tokens)])
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=CustomDataset.collate_fn
    )

    batch = next(iter(loader))
    with torch.no_grad():
        risk_outputs, priority_outputs = model(batch.tokens, batch.mask)

    risk_probs = risk_outputs[0][:batch.lengths[0] - 1].cpu().tolist()
    priority_scores = priority_outputs[0][:batch.lengths[0] - 1].cpu().tolist()

    risk_preds = [1 if x >= 0.5 else 0 for x in risk_probs]

    def transform(prediction_sent, ids_sent):
        mapping = {i: None for i in range(max(ids_sent) + 1)}
        for prediction_word, index in zip(prediction_sent, ids_sent):
            if mapping[index] is None:
                mapping[index] = prediction_word
        return [mapping[i] for i in range(max(ids_sent) + 1)]

    risk_preds = transform(risk_preds, batch.mapping[0])
    priority_scores = transform(priority_scores, batch.mapping[0])

    return risk_preds, priority_scores
