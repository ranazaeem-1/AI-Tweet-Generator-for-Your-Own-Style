import argparse
from pathlib import Path
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

BASE_DIR = Path(__file__).resolve().parents[1]
MAX_CHAR_LEN = 280


def load_generator(model_path: str):
    """Load model and tokenizer, with helpful error messages."""
    model_path_obj = Path(model_path)
    
    required_files = [
        model_path_obj / "pytorch_model.bin",
        model_path_obj / "config.json",
        model_path_obj / "tokenizer_config.json",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Model files missing from {model_path}:\n"
            f"  Missing: {[f.name for f in missing]}\n"
            f"  Run: python src/train_model.py"
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def trim_tweet(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:MAX_CHAR_LEN]


def generate_tweets(
    generator,
    prompt: str,
    num_tweets: int = 3,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 2,
) -> List[str]:
    outputs = generator(
        prompt,
        num_return_sequences=num_tweets,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        return_full_text=False,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [trim_tweet(o["generated_text"]) for o in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tweets using a fine-tuned GPT-2 model.")
    parser.add_argument("--prompt", required=True, help="Seed text for generation.")
    parser.add_argument("--model-path", default=str(BASE_DIR / "model" / "fine_tuned_model"), help="Path to fine-tuned model directory.")
    parser.add_argument("--num-tweets", type=int, default=3, help="Number of tweets to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling cutoff.")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty to discourage repetition.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2, help="Disallow repeating n-grams of this size.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Max new tokens to generate.")
    parser.add_argument("--seed", type=int, help="Optional random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    generator = load_generator(args.model_path)
    tweets = generate_tweets(
        generator=generator,
        prompt=args.prompt,
        num_tweets=args.num_tweets,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print(f"Prompt: {args.prompt}")
    for idx, tweet in enumerate(tweets, start=1):
        print(f"[{idx}] {tweet}")


if __name__ == "__main__":
    main()
