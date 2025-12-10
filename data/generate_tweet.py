import argparse
from pathlib import Path
from typing import List

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


def load_generator(model_path: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def trim_tweet(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:280]


def generate_tweets(
    generator,
    prompt: str,
    num_tweets: int = 3,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> List[str]:
    outputs = generator(
        prompt,
        num_return_sequences=num_tweets,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        return_full_text=False,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [trim_tweet(o["generated_text"]) for o in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tweets using a fine-tuned GPT-2 model.")
    parser.add_argument("--prompt", required=True, help="Seed text for generation.")
    parser.add_argument("--model-path", default="model/fine_tuned_model", help="Path to fine-tuned model directory.")
    parser.add_argument("--num-tweets", type=int, default=3, help="Number of tweets to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
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
    )

    print(f"Prompt: {args.prompt}")
    for idx, tweet in enumerate(tweets, start=1):
        print(f"[{idx}] {tweet}")


if __name__ == "__main__":
    main()
