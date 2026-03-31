# Ticket Risk RAG Assistant

A business-oriented NLP project for customer service ticket risk detection and response assistance.

## Project Goal

This project transforms an old research prototype for metaphor and novelty prediction into a practical AI application for customer service and operations teams.

The current version keeps the original dual-head architecture, but changes the project language and exported outputs to a business scenario:

- token-level risk phrase detection
- token-level priority scoring
- prediction export for downstream workflow integration

## Current Stage

This package is a **drop-in step-2 version** of the old codebase:

- project wording changed from metaphor/novelty to risk/priority
- `transformers.AdamW` compatibility issue fixed by using `torch.optim.AdamW`
- inference CSV reader bug fixed (`next(corpus)` instead of `corpus.readline()`)
- project structure updated to `src/`

## Project Structure

```text
ticket-risk-step2-ready/
├── README.md
├── main.py
├── predict.py
├── requirements.txt
├── data/
│   └── ticket_samples.csv
├── knowledge_base/
│   └── refund_policy.md
└── src/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── model.py
    └── trainer.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py --seed 1 --lr 3e-5 --train_steps 1000 --batch_size 32 --alpha 0.5 --beta 0.5 --metaphor_weight 0.8
```

## Prediction

```bash
python predict.py --model_name model.pt --output ticket_predictions.tsv
```

## Notes

This is still a transitional version:
- internal tensor field names in `src/data.py` are kept for compatibility
- the next step is to replace the original VUA data format with real ticket-risk data
- the next business step is to add RAG retrieval and an API layer
