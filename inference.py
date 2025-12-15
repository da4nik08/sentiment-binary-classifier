import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer
from models.mini_lm_classifier import MiniLMSentimentClassifier
from utils.custom_dataset import TokenizedTextDataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from utils.load_cfg import load_config


def load_model(cfg):
    device = ("cuda" if torch.cuda.is_available() and cfg["model"]["device"] == "auto" else cfg["model"]["device"])
    model = MiniLMSentimentClassifier(model_name=cfg["model"]["name"], device=device)
    checkpoint = torch.load(cfg["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def run_inference(texts: list[str], model: MiniLMSentimentClassifier, tokenizer, cfg: dict) -> list[dict]:

    encoded = tokenizer(
        texts,
        padding=cfg["tokenizer"]["padding"],
        truncation=cfg["tokenizer"]["truncation"],
        max_length=cfg["tokenizer"]["max_length"],
        return_tensors="pt"
    )

    dummy_labels = torch.zeros(len(texts))  # того ж розміру що і звичайні проте не по
    dataset = TokenizedTextDataset(encoded, dummy_labels)
    loader = DataLoader(
        dataset,
        batch_size=cfg["inference"]["batch_size"],
        shuffle=False
    )

    results = []
    threshold = cfg["inference"]["threshold"]

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            token_type_ids = (
                batch["token_type_ids"].to(model.device)
                if "token_type_ids" in batch else None
            )

            logits = model(input_ids, token_type_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(1)

            preds = (probs >= threshold).long()

            for p, pr in zip(preds.cpu().numpy(), probs.cpu().numpy()):
                results.append({
                    "prediction": int(p),
                    "probability": float(pr)
                })

    return results


def predict_from_csv(csv_path: str, text_column: str, config_path: str):

    cfg = load_config(config_path)
    model = load_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    df = pd.read_csv(csv_path)
    texts = df[text_column].astype(str).tolist()
    preds = run_inference(texts, model, tokenizer, cfg)

    df["prediction"] = [p["prediction"] for p in preds]
    df["probability"] = [p["probability"] for p in preds]
    return df


if __name__ == "__main__":
    output = predict_from_csv(
        csv_path="dataset/test.csv",
        text_column="review_final",
        config_path="configs/inference_config.yaml"
    )
    output.to_csv("predictions.csv", index=False)

