
import pandas as pd
import spacy
import re
import os
import yaml

nlp = spacy.load("en_core_web_sm")

def preprocess_text(comment: str) -> str:
    comment = comment.lower().strip()
    comment = re.sub(r'\\\\n|\\n', ' ', comment)
    comment = re.sub(r'http\\S+|www\\S+', '', comment)
    comment = re.sub(r'@\\w+', '', comment)
    comment = re.sub(r'#\\w+', '', comment)
    comment = re.sub(r'[^\\w\\s]', '', comment)
    comment = re.sub(r'\\s+', ' ', comment).strip()
    
    doc = nlp(comment)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    train_data = pd.read_csv(os.path.join(raw_dir, config["data"]["train_file"])).iloc[:3000, :]
    train_data["comment_text_clean"] = train_data["comment_text"].apply(preprocess_text)
    train_data.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)

if __name__ == "__main__":
    main()