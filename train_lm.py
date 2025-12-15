import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from utils.custom_dataset import TokenizedTextDataset
from utils.metrics import MetricsWriter
from sklearn.model_selection import train_test_split
from models.minilm_classifier import MiniLMSentimentClassifier
from training_funtions.train_lm_loop import train
from utils.load_cfg import load_config
from pathlib import Path


def train_model(config_path):
    cfg = load_config(config_path)
    
    dataset_path = Path(cfg["dataset"]["path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {dataset_path}")
        
    dataset = pd.read_csv(dataset_path)
    data = dataset.copy()
    X = df[cfg["dataset"]["text_column"]].tolist()
    y = df[cfg["dataset"]["label_column"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["dataset"]["test_size"],
                                                              random_state=cfg["dataset"]["random_state"], stratify=y)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    encoded_input_train = tokenizer(X_train, padding=True, truncation=True,
                              max_length=cfg["tokenizer"]["max_length"], return_tensors='pt')
    encoded_input_test = tokenizer(X_test, padding=True, truncation=True,
                              max_length=cfg["tokenizer"]["max_length"], return_tensors='pt')

    train_dataset = TokenizedTextDataset(encoded_input_train, y_train)
    test_dataset = TokenizedTextDataset(encoded_input_test, y_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], 
                                                   num_workers=cfg["training"]["num_workers"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], 
                                                 num_workers=cfg["training"]["num_workers"])

    device = ("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "auto" else "cpu")
    model = MiniLMSentimentClassifier(
        model_name=cfg["model"]["name"],
        dropout_p=cfg["model"]["dropout"],
        device=device
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    for p in model.encoder.parameters():
        p.requires_grad = False

    n = cfg["model"]["unfreeze_last_n_layers"]
    for layer in model.encoder.encoder.layer[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    no_decay = ["bias", "LayerNorm.weight"]
    encoder_decay = []
    encoder_no_decay = []
    
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue                      # скіп заморожених
        if any(nd in name for nd in no_decay):
            encoder_no_decay.append(param)
        else:
            encoder_decay.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_decay, "lr": cfg["optimizer"]["encoder_lr"], "weight_decay": cfg["optimizer"]["weight_decay"]},
        {"params": encoder_no_decay, "lr": cfg["optimizer"]["encoder_lr"], "weight_decay": 0.0},
        {"params": model.fc1.parameters(), "lr": "lr": cfg["optimizer"]["head_lr"], "weight_decay": cfg["optimizer"]["weight_decay"]},
    ])
    
    mw = MetricsWriter(model, model_name=cfg["logging"]["run_name"], save_treshold=cfg["logging"]["save_threshold"])
    train(model, mw, loss_fn, optimizer, train_dataloader, val_dataloader, epochs=cfg["training"]["epochs"])