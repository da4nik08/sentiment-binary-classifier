import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import mlflow
import re
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_review(text: str) -> str:
    text = text.lower()                
    text = re.sub(r"<.*?>", " ", text) 
    text = text.replace("\\'", "'").replace('\\"', '"').replace("\\n", " ")  # Прибираємо escape-послідовності типу \', \", \\n

    emoji_pattern = re.compile(  #  Прибираємо смайлики та юнікодні emoji
        "["                   
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002700-\U000027BF"  
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    text = text.replace("’", "'").replace("‘", "'") # нормалізуємо лапки

    text = re.sub(r"(?<![a-z0-9])'(?![a-z0-9])", " ", text)  # видаляємо апострофи, що НЕ між літерами/цифрами
    text = re.sub(r"\s+'(?=[a-z0-9])", " ", text)   # 'word → word
    text = re.sub(r"(?<=[a-z0-9])'\s+", " ", text)  # word' → word
    
    text = re.sub(r"[^a-z0-9\s'\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()        # Видаляємо повторювані пробіли

    return text

def remove_stopwords(text: str) -> str:
        words = text.split()
        filtered_words = [w for w in words if w not in stop_words]
        return " ".join(filtered_words)

def trim_review(text, max_words=400):
    words = text.split()
    if len(words) <= max_words:
        return text
    if len(words) > (max_words + 50): # якщо . ! ? немає
        return " ".join(words[:max_words])
    
    trimmed = words[:max_words]
    for w in words[max_words:]:
        trimmed.append(w)
        if re.search(r'[.!?]$', w):
            break
    return " ".join(trimmed)

def preprocess_dataset(dataset_path, result_ds_name):

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {dataset_path}")
        
    dataset = pd.read_csv(dataset_path)
    data = dataset.copy()
    data["review_clean"] = data["review"].apply(clean_review)
    data["review_final"] = data["review_clean"].apply(remove_stopwords)
    data['label'] = data['sentiment'].map({'negative': 0, 'positive': 1})
    data['review_final'] = data['review_final'].apply(trim_review)

    processed_path = Path("dataset") / str(result_ds_name)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)
    print(f"[INFO] Датасет успішно збережено: {processed_path}")