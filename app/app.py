from pathlib import Path
from typing import List

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "fine_tuned_model"


@st.cache_resource
def load_generator(model_path: Path = MODEL_DIR):
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
    num_tweets: int,
    temperature: float,
    max_new_tokens: int,
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


def main() -> None:
    st.title("AI Tweet Generator")
    st.write("Generate tweets in your style using the fine-tuned GPT-2 model.")

    prompt = st.text_input("Enter a tweet prompt:")
    num_tweets = st.slider("Number of tweets", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.2, 1.5, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 20, 120, 80, 5)
    seed = st.text_input("Optional seed (integer)", value="")

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please enter a prompt to generate tweets.")
            return

        if seed.strip():
            try:
                set_seed(int(seed.strip()))
            except ValueError:
                st.warning("Seed must be an integer. Using random seed.")

        try:
            generator = load_generator()
            tweets = generate_tweets(
                generator=generator,
                prompt=prompt,
                num_tweets=num_tweets,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            for idx, tweet in enumerate(tweets, start=1):
                st.markdown(f"**Tweet {idx}:** {tweet}")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
