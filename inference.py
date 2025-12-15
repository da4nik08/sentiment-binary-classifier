import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer
from models.minilm_classifier import MiniLMSentimentClassifier
from utils.custom_dataset import TokenizedTextDataset
from preprocess import preprocess_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
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

    dummy_labels = torch.zeros(len(texts))  # того ж розміру що і звичайні не використовується далі 
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

    preprocess_dataset(csv_path, 'temp_dataset')
    processed_path = Path("dataset") / str('temp_dataset')
    df = pd.read_csv(processed_path)
    texts = df[text_column].astype(str).tolist()
    preds = run_inference(texts, model, tokenizer, cfg)

    df["prediction"] = [p["prediction"] for p in preds]
    df["probability"] = [p["probability"] for p in preds]
    return df


def main():
    parser = argparse.ArgumentParser(description="Run sentiment inference on CSV dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--text_column", type=str, default=None, help="Name of the text column")
    parser.add_argument("--config_path", type=str, default="configs/inference_config.yaml", help="Path to config YAML")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="Path to save predictions CSV")

    args = parser.parse_args()
    cfg = load_config(args.config_path)

    text_column = args.text_column or cfg.get("inference", {}).get("text_column", "review_final")

    df = predict_from_csv(args.csv_path, text_column, args.config_path)
    df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()