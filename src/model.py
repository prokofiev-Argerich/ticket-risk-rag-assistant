"""
Models for the ticket risk project.
Create the neural network that we will use for risk prediction.

- BERT:
    - BERT-base-cased
    - a classification layer for token-level risk prediction with sigmoid
    - a regression layer for token-level priority scoring with tanh
"""
import torch

from transformers import BertModel


class MetaphorModel(torch.nn.Module):
    """Risk detection model."""

    def __init__(self):
        super().__init__()
        self.name = "model=bert"

        self.bert_model = BertModel.from_pretrained("bert-base-cased")
        self.bert_model.train()

        self.dropout_output = torch.nn.Dropout(0.1)
        self.risk_output_projection = torch.nn.Linear(768, 1)
        self.priority_output_projection = torch.nn.Linear(768, 1)

    def forward(self, inputs, mask):
        """
        Forward pass of BERT model for ticket risk prediction.

        Args:
            inputs: token ids
            mask: PyTorch LongTensor of zeros and ones

        Returns:
            risk_scores: batch_size x max_sent_length of float predictions
            priority_scores: batch_size x max_sent_length of float predictions
        """
        if not torch.cuda.is_available():
            bert_output = self.bert_model(
                input_ids=inputs, attention_mask=mask
            )[0]
        else:
            bert_output = self.bert_model(
                input_ids=inputs.cuda(), attention_mask=mask.cuda()
            )[0]

        encoded_sentence = self.dropout_output(bert_output)

        risk_scores = torch.sigmoid(
            self.risk_output_projection(encoded_sentence)
        ).squeeze(-1)

        priority_scores = torch.tanh(
            self.priority_output_projection(encoded_sentence)
        ).squeeze(-1)

        return risk_scores[:, 1:-1], priority_scores[:, 1:-1]
