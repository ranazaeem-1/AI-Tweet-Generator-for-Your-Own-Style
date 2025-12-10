import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on tweet data.")
    parser.add_argument("--train-data", default=str(BASE_DIR / "data" / "processed" / "train_data"), help="Path to train dataset saved with save_to_disk.")
    parser.add_argument("--test-data", default=str(BASE_DIR / "data" / "processed" / "test_data"), help="Path to test dataset saved with save_to_disk.")
    parser.add_argument("--tokenizer", default=str(BASE_DIR / "data" / "processed" / "tokenizer"), help="Tokenizer path (from preprocess_data).")
    parser.add_argument("--model-name", default="distilgpt2", help="Base model checkpoint.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "model" / "fine_tuned_model"), help="Directory to store fine-tuned model.")
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

    # ===== VALIDATION =====
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    tokenizer_path = Path(args.tokenizer)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {args.train_data}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {args.test_data}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    
    print(f"📂 Loading preprocessed data from {args.train_data}")
    
    # ===== LOAD DATA & MODEL =====
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_from_disk(args.train_data)
    test_dataset = load_from_disk(args.test_data)
    
    print(f"📊 Dataset Info:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Run preprocessing first.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Run preprocessing first.")

    print(f"🤖 Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ===== TRAINING CONFIG =====
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
        logging_dir='./logs',
    )

    print(f"⚙️ Training Config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Batch Size: {args.train_batch_size}")
    print(f"   - FP16: {args.fp16}")
    print(f"   - Output Dir: {args.output_dir}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ===== TRAINING =====
    print(f"🚀 Starting training...")
    trainer.train()
    
    # ===== SAVE =====
    output_path = Path(args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Training complete!")
    print(f"📦 Model saved to {output_path.resolve()}")
    print(f"🎉 Ready for generation! Use: python src/generate_tweet.py --prompt 'YOUR_PROMPT'")


if __name__ == "__main__":
    main()
