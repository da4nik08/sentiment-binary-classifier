import torch
from torch.utils.data import Dataset


class TokenizedTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return { "input_ids": self.encodings["input_ids"][idx], 
                "attention_mask": self.encodings["attention_mask"][idx], 
                "token_type_ids": self.encodings["token_type_ids"][idx], 
                "labels": torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)}