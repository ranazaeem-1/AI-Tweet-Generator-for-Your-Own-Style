import argparse
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer


def remove_noise(text: str) -> str:
    url_pattern = re.compile(r"http\S+|www\.\S+")
    mention_pattern = re.compile(r"@\w+")
    hashtag_pattern = re.compile(r"#(\w+)")

    cleaned = url_pattern.sub("", text)
    cleaned = mention_pattern.sub("", cleaned)
    cleaned = hashtag_pattern.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tweet data for GPT-2 fine-tuning.")
    parser.add_argument("--input", default="data/tweets_data.csv", help="CSV file containing tweet_text,date.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to store tokenized datasets.")
    parser.add_argument("--model-name", default="gpt2", help="Tokenizer model name or local path.")
    parser.add_argument("--max-length", type=int, default=280, help="Max token length per tweet.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--clean", action="store_true", help="Clean tweets by removing URLs and mentions.")
    parser.add_argument("--separator", default="<|endoftext|>", help="Separator appended to each tweet before tokenizing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "tweet_text" not in df.columns:
        raise ValueError("Input CSV must contain a tweet_text column.")

    if args.clean:
        df["tweet_text"] = df["tweet_text"].astype(str).map(remove_noise)
    df = df.dropna(subset=["tweet_text"])

    dataset = Dataset.from_pandas(df[["tweet_text"]], preserve_index=False)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    sep = args.separator if args.separator else tokenizer.eos_token

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [f"{text} {sep}" for text in examples["tweet_text"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["tweet_text"])

    split = tokenized.train_test_split(test_size=args.test_size, seed=args.seed)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train_data"
    test_dir = output_dir / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    split["train"].save_to_disk(train_dir)
    split["test"].save_to_disk(test_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print(f"Saved train dataset to {train_dir}")
    print(f"Saved test dataset to {test_dir}")
    print(f"Saved tokenizer to {output_dir / 'tokenizer'}")


if __name__ == "__main__":
    main()
