import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on tweet data.")
    parser.add_argument("--train-data", default="data/processed/train_data", help="Path to train dataset saved with save_to_disk.")
    parser.add_argument("--test-data", default="data/processed/test_data", help="Path to test dataset saved with save_to_disk.")
    parser.add_argument("--tokenizer", default="data/processed/tokenizer", help="Tokenizer path (from preprocess_data).")
    parser.add_argument("--model-name", default="distilgpt2", help="Base model checkpoint.")
    parser.add_argument("--output-dir", default="model/fine_tuned_model", help="Directory to store fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Per-device batch size for training.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Per-device batch size for evaluation.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Steps between logging calls.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training (if supported).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer or args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_from_disk(args.train_data)
    test_dataset = load_from_disk(args.test_data)

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
