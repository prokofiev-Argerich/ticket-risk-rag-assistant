import json
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")


class Batch:
    def __init__(
        self,
        sentences,
        lengths,
        mask,
        tokens,
        mapping,
        bert_risk_labels,
        bert_priority_labels,
        risk_labels,
        priority_labels,
    ):
        self.sentences = sentences
        self.lengths = torch.LongTensor(lengths)
        self.mask = torch.FloatTensor(mask)

        self.mapping = mapping
        self.tokens = torch.LongTensor(tokens)
        self.bert_metaphoricity_labels = torch.FloatTensor(bert_risk_labels)
        self.bert_novelty_labels = torch.FloatTensor(bert_priority_labels)
        self.metaphoricity_labels = risk_labels
        self.novelty_labels = priority_labels


class CustomDataset(Dataset):
    def __init__(self, texts, risk_labels, priority_labels):
        self.texts = texts
        self.risk_labels = risk_labels
        self.priority_labels = priority_labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        risk_label = self.risk_labels[idx]
        priority_label = self.priority_labels[idx]
        return text, risk_label, priority_label

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def collate_fn(batch):
        def bert_process_sentences(sentences, labels):
            bert_sentences, bert_lengths, bert_labels = [], [], []
            mapping = []

            for sentence, label_seq in zip(sentences, labels):
                bert_label = []
                mapping_list = []
                bert_sentence = TOKENIZER.convert_tokens_to_ids(["[CLS]"])

                for i, (word, word_label) in enumerate(zip(sentence, label_seq)):
                    word_pieces = TOKENIZER.tokenize(word)
                    word_piece_ids = TOKENIZER.convert_tokens_to_ids(word_pieces)

                    bert_sentence.extend(word_piece_ids)
                    bert_label.extend([word_label] * len(word_piece_ids))
                    mapping_list.extend([i] * len(word_piece_ids))

                mapping.append(mapping_list)
                bert_sentences.append(
                    bert_sentence + TOKENIZER.convert_tokens_to_ids(["[SEP]"])
                )
                bert_labels.append(bert_label)
                bert_lengths.append(len(bert_sentence) + 1)

            max_len = max(len(s) for s in bert_sentences)

            tokens, mask = [], []
            for i, sent in enumerate(bert_sentences):
                mask.append([1] * len(sent) + [0] * (max_len - len(sent)))
                tokens.append(sent + [0] * (max_len - len(sent)))
                bert_labels[i] = bert_labels[i] + [-2] * (max_len - len(sent))

            return tokens, bert_lengths, mask, bert_labels, mapping

        sentences, risk_labels, priority_labels = zip(*batch)

        bert_tokens, bert_lengths, bert_mask, bert_risk_labels, mapping = \
            bert_process_sentences(sentences, risk_labels)

        _, _, _, bert_priority_labels, _ = \
            bert_process_sentences(sentences, priority_labels)

        return Batch(
            sentences,
            bert_lengths,
            bert_mask,
            bert_tokens,
            mapping,
            bert_risk_labels,
            bert_priority_labels,
            risk_labels,
            priority_labels,
        )


def load_jsonl_dataset(path):
    texts, risk_labels, priority_labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = row["tokens"]
            risk = row["risk_labels"]
            priority = row["priority_scores"]

            assert len(tokens) == len(risk) == len(priority)

            texts.append(tokens)
            risk_labels.append(risk)
            priority_labels.append(priority)

    return texts, risk_labels, priority_labels


def build_dataloader(path, train=False, batch_size=32):
    texts, risk_labels, priority_labels = load_jsonl_dataset(path)
    dataset = CustomDataset(texts, risk_labels, priority_labels)
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=CustomDataset.collate_fn,
    )


def load_data(batch_size=32):
    train_loader = build_dataloader(
        "data/ticket_risk_train.jsonl", train=True, batch_size=batch_size
    )
    dev_loader = build_dataloader(
        "data/ticket_risk_dev.jsonl", train=False, batch_size=batch_size
    )
    test_loader = build_dataloader(
        "data/ticket_risk_test.jsonl", train=False, batch_size=batch_size
    )
    return train_loader, dev_loader, test_loader
