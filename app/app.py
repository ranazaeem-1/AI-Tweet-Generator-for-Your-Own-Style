from pathlib import Path
from typing import List, Tuple

import streamlit as st
import tweepy
from tweepy import TweepyException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "fine_tuned_model"
MAX_CHAR_LEN = 280
STYLE_TEMPS = {"Witty": 0.9, "Serious": 0.5, "Casual": 0.7}
DEFAULT_FETCH_MAX = 3000
MIN_FETCH = 50
MONTHLY_POST_LIMIT = 100


@st.cache_resource
def load_generator(model_path: Path = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def build_twitter_api(
    api_key: str, api_secret: str, access_token: str, access_secret: str
) -> Tuple[tweepy.API, tweepy.User]:
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    user = api.verify_credentials()
    if not user:
        raise TweepyException("Authentication failed; verify credentials.")
    return api, user


def fetch_user_tweets(
    api: tweepy.API,
    screen_name: str,
    max_items: int = DEFAULT_FETCH_MAX,
    include_rts: bool = False,
    exclude_replies: bool = True,
) -> List[dict]:
    target_items = max(max_items, MIN_FETCH)
    cursor = tweepy.Cursor(
        api.user_timeline,
        screen_name=screen_name,
        tweet_mode="extended",
        include_rts=include_rts,
        exclude_replies=exclude_replies,
        count=200,
    )
    tweets: List[dict] = []
    try:
        for tweet in cursor.items(target_items):
            tweets.append(
                {
                    "tweet_text": tweet.full_text.replace("\n", " ").strip(),
                    "date": tweet.created_at.isoformat(),
                    "id": tweet.id_str,
                }
            )
    except TweepyException as exc:
        raise exc
    return tweets


def trim_tweet(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:MAX_CHAR_LEN]


def generate_tweets(
    generator,
    prompt: str,
    num_tweets: int,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
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


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest" not in st.session_state:
        st.session_state.latest = []
    if "fetched_tweets" not in st.session_state:
        st.session_state.fetched_tweets = []
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None


def main() -> None:
    init_state()
    st.title("AI Tweet Generator")
    st.write("Generate tweets in your style using the fine-tuned GPT-2 model.")

    st.header("Twitter API Credentials")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret Key", type="password")
    access_token = st.text_input("Access Token", type="password")
    access_secret = st.text_input("Access Token Secret", type="password")
    handle = st.text_input("Twitter handle to fetch tweets (without @)")
    include_rts = st.checkbox("Include retweets", value=False)
    include_replies = st.checkbox("Include replies", value=False)
    fetch_limit = st.slider("Max tweets to fetch", MIN_FETCH, DEFAULT_FETCH_MAX, DEFAULT_FETCH_MAX, 50)

    if st.button("Fetch Tweets"):
        if not (api_key and api_secret and access_token and access_secret and handle.strip()):
            st.warning("Please enter all Twitter API credentials and a handle to fetch tweets.")
        else:
            try:
                api, user = build_twitter_api(api_key, api_secret, access_token, access_secret)
                st.session_state.auth_user = user.screen_name
                st.success(f"Authentication successful for @{user.screen_name}")
                with st.spinner("Fetching tweets..."):
                    tweets = fetch_user_tweets(
                        api=api,
                        screen_name=handle.strip(),
                        max_items=fetch_limit,
                        include_rts=include_rts,
                        exclude_replies=not include_replies,
                    )
                st.session_state.fetched_tweets = tweets
                if tweets:
                    st.success(f"Fetched {len(tweets)} tweets.")
                    if len(tweets) < MIN_FETCH:
                        st.info(f"Only {len(tweets)} tweets available for this account (requested at least {MIN_FETCH}).")
                    st.info(f"Reminder: Free tiers often cap timeline reads to ~{MONTHLY_POST_LIMIT} posts per month.")
                else:
                    st.warning(
                        "No tweets returned. Check if the account has tweets, "
                        "if access is limited (free tier often caps at 100 posts/month), "
                        "or if timeline visibility is restricted."
                    )
            except TweepyException as exc:
                code = getattr(exc, "api_code", None)
                if code == 88:
                    st.error("Rate limit exceeded. Please wait and try again later.")
                elif code in (453, 403):
                    st.error(
                        "Access forbidden or limited. This endpoint requires Basic/Pro/Elevated access. "
                        "Upgrade your X developer plan to read timelines: https://developer.x.com/en/portal/product"
                    )
                else:
                    st.error(f"Failed to fetch tweets: {exc}")
            except Exception as exc:
                st.error(f"Failed to fetch tweets: {exc}")

    if st.session_state.fetched_tweets:
        st.subheader(f"Fetched tweets preview (first 5 of {len(st.session_state.fetched_tweets)})")
        for tweet in st.session_state.fetched_tweets[:5]:
            st.write(f"- {tweet['tweet_text']}")
    elif handle.strip():
        st.info("No tweets fetched yet. Ensure credentials are correct and fetch to preview.")

    st.header("Generate Tweets")
    prompt = st.text_input("Enter a tweet prompt:")
    style = st.selectbox("Choose tweet style:", list(STYLE_TEMPS.keys()))
    num_tweets = st.slider("Number of tweets", 1, 5, 3)
    max_new_tokens = st.slider("Max new tokens", 20, 120, 80, 5)
    top_k = st.slider("Top-k", 10, 200, 50, 5)
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.95, 0.01)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
    no_repeat_ngram_size = st.slider("No-repeat n-gram size", 1, 4, 2)
    seed = st.text_input("Optional seed (integer)", value="")

    if st.button("Generate Tweets"):
        prompt_clean = prompt.strip()
        if len(prompt_clean) < 3:
            st.warning("Please enter a longer prompt to generate tweets.")
            return

        if seed.strip():
            try:
                set_seed(int(seed.strip()))
            except ValueError:
                st.warning("Seed must be an integer. Using random seed.")

        try:
            generator = load_generator()
            with st.spinner("Generating tweets..."):
                tweets = generate_tweets(
                    generator=generator,
                    prompt=prompt_clean,
                    num_tweets=num_tweets,
                    temperature=STYLE_TEMPS[style],
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            st.session_state.latest = tweets

            for idx, tweet in enumerate(tweets, start=1):
                st.markdown(f"**Tweet {idx} ({style}):** {tweet}")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    if st.session_state.latest:
        if st.button("Save latest tweets to history"):
            st.session_state.history.extend(st.session_state.latest)
            st.success("Saved latest tweets to history.")

    st.subheader("Tweet History")
    if st.session_state.history:
        for idx, tweet in enumerate(st.session_state.history, start=1):
            st.markdown(f"{idx}. {tweet}")
        if st.button("Clear history"):
            st.session_state.history.clear()
            st.info("History cleared.")
    else:
        st.write("No tweets in history yet.")


if __name__ == "__main__":
    main()
