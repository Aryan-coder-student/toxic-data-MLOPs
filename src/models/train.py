# src/models/train.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
import os

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load data
    processed_dir = config["data"]["processed_dir"]
    train_data = pd.read_csv(os.path.join(processed_dir, "train_processed.csv"))
    X = train_data["comment_text_clean"].values
    y = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, config["model"]["max_len"])
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer, config["model"]["max_len"])

    # Model
    model = BertForSequenceClassification.from_pretrained(
        config["model"]["name"], num_labels=config["model"]["num_labels"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["model"]["output_dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        learning_rate=config["training"]["learning_rate"],
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()

    # Save the model
    model.save_pretrained(config["model"]["output_dir"])
    tokenizer.save_pretrained(config["model"]["output_dir"])

if __name__ == "__main__":
    main()