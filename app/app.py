from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

from dotenv import load_dotenv
import streamlit as st
import tweepy
from tweepy import TweepyException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

# Load environment variables from .env file
load_dotenv()

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "fine_tuned_model"
MAX_CHAR_LEN = 280
STYLE_TEMPS = {"Witty": 0.9, "Serious": 0.5, "Casual": 0.7}
# Free tier limits for X API (as of 2025)
DEFAULT_FETCH_MAX = 100  # Free tier: max 100 posts/month
MIN_FETCH = 20  # Realistic minimum for quality model
MAX_FREE_TIER_POSTS = 50  # Posts per month
MAX_FREE_TIER_READS = 100  # Reads per month


# ============= VALIDATION & UTILITIES =============

def check_model_exists() -> bool:
    """Verify model files exist before starting app."""
    required_files = [
        MODEL_DIR / "pytorch_model.bin",
        MODEL_DIR / "config.json",
        MODEL_DIR / "tokenizer_config.json",
    ]
    return all(f.exists() for f in required_files)


def validate_api_credentials(api_key: str, api_secret: str, access_token: str, access_secret: str) -> Tuple[bool, str]:
    """Validate credentials by attempting authentication."""
    if not all([api_key, api_secret, access_token, access_secret]):
        return False, "Missing one or more credentials."
    
    try:
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api = tweepy.API(auth)
        user = api.verify_credentials()
        return True, f"✅ Authenticated as @{user.screen_name}"
    except TweepyException as e:
        return False, f"❌ Authentication failed: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ============= CORE FUNCTIONS =============

@st.cache_resource
def load_generator(model_path: Path = MODEL_DIR):
    """Load fine-tuned model and tokenizer for text generation."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"❌ Failed to load model from {model_path}: {e}")
        st.info("💡 Make sure to run training first: `python src/train_model.py`")
        st.stop()


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
    """Fetch tweets from user timeline with free-tier safety limits."""
    # Cap at free tier limit
    target_items = max(min(max_items, MAX_FREE_TIER_READS), MIN_FETCH)
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
    """Initialize Streamlit session state."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest" not in st.session_state:
        st.session_state.latest = []
    if "fetched_tweets" not in st.session_state:
        st.session_state.fetched_tweets = []
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "credentials_valid" not in st.session_state:
        st.session_state.credentials_valid = False


def get_free_tier_warning() -> str:
    """Return a warning message about free tier limits."""
    return f"""
    ⚠️ **X Free Tier Limits:**
    - Max tweets to fetch per month: {MAX_FREE_TIER_READS}
    - Max tweets to post per month: {MAX_FREE_TIER_POSTS}
    - Upgrade to Basic+ tier for higher limits: https://developer.x.com/en/portal/product
    """


def handle_api_error(exc: TweepyException) -> str:
    """Convert Tweepy errors to user-friendly messages."""
    code = getattr(exc, "api_code", None)
    
    if code == 88:
        return "🔄 **Rate limit exceeded.** Free tier allows ~100 posts/month. Please try again later."
    elif code == 403:
        return "🔐 **Access forbidden.** You may need Basic/Pro/Elevated access for this endpoint."
    elif code == 453:
        return "🔐 **Access limited.** Upgrade your X developer tier for timeline access."
    elif code == 429:
        return "⏱️ **Too many requests.** Wait a moment before trying again."
    elif code == 401:
        return "🔑 **Unauthorized.** Your credentials may be invalid. Please re-enter them."
    else:
        return f"❌ **API Error ({code}):** {str(exc)}"


def main() -> None:
    init_state()
    
    # Page configuration
    st.set_page_config(
        page_title="AI Tweet Generator",
        page_icon="🐦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🐦 AI Tweet Generator")
    st.markdown(
        "Generate tweets in **your unique style** using a fine-tuned GPT-2 model. "
        "Train on your past tweets and create new ones with custom prompts!"
    )
    
    # Integrated pipeline state
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = "idle"
    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = False
    if "trained" not in st.session_state:
        st.session_state.trained = check_model_exists()
    if "train_log" not in st.session_state:
        st.session_state.train_log = ""
    
    # Sidebar - API Credentials
    with st.sidebar:
        st.header("🔐 Twitter API Setup")
        st.markdown(
            "[Get credentials from X Developer Portal](https://developer.x.com/en/portal/dashboard)"
        )
        
        # Load from .env file if available
        env_api_key = os.getenv("TWITTER_API_KEY", "")
        env_api_secret = os.getenv("TWITTER_API_KEY_SECRET", "")
        env_access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        env_access_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
        
        st.info("💡 **Tip:** Add credentials to `.env` file to auto-load them!")
        
        with st.expander("ℹ️ Free Tier Info", expanded=False):
            st.markdown(get_free_tier_warning())
        
        api_key = st.text_input(
            "API Key", 
            value=env_api_key,
            type="password", 
            key="api_key"
        )
        api_secret = st.text_input(
            "API Secret Key", 
            value=env_api_secret,
            type="password", 
            key="api_secret"
        )
        access_token = st.text_input(
            "Access Token", 
            value=env_access_token,
            type="password", 
            key="access_token"
        )
        access_secret = st.text_input(
            "Access Token Secret", 
            value=env_access_secret,
            type="password", 
            key="access_secret"
        )
        
        if st.button("✅ Verify Credentials", use_container_width=True):
            is_valid, message = validate_api_credentials(api_key, api_secret, access_token, access_secret)
            st.session_state.credentials_valid = is_valid
            if is_valid:
                st.success(message)
            else:
                st.error(message)
    
    # Main content area
    handle = st.text_input("🐦 Twitter handle to fetch tweets (without @)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        include_rts = st.checkbox("Include retweets", value=False)
    with col2:
        include_replies = st.checkbox("Include replies", value=False)
    with col3:
        fetch_limit = st.slider("Max tweets to fetch", MIN_FETCH, DEFAULT_FETCH_MAX, DEFAULT_FETCH_MAX, 10)
    
    # Integrated pipeline UI
    st.subheader("🚀 Full Workflow: Fetch → Preprocess → Train → Generate")
    st.markdown("Run the entire pipeline from here. No manual script execution required!")

    # Step 1: Fetch Tweets
    fetch_btn = st.button("1️⃣ Fetch Tweets", use_container_width=True)
    if fetch_btn:
        st.session_state.pipeline_status = "fetching"
        try:
            api, user = build_twitter_api(api_key, api_secret, access_token, access_secret)
            st.session_state.auth_user = user.screen_name
            with st.spinner("🔄 Fetching tweets..."):
                tweets = fetch_user_tweets(
                    api=api,
                    screen_name=handle.strip(),
                    max_items=fetch_limit,
                    include_rts=include_rts,
                    exclude_replies=not include_replies,
                )
            st.session_state.fetched_tweets = tweets
            if tweets:
                st.success(f"✅ Fetched {len(tweets)} tweets.")
            else:
                st.warning("⚠️ No tweets returned. Check account or API limits.")
        except TweepyException as exc:
            st.error(handle_api_error(exc))
        except Exception as exc:
            st.error(f"❌ Error: {str(exc)}")
        st.session_state.pipeline_status = "idle"

    # Step 2: Preprocess Tweets
    preprocess_btn = st.button("2️⃣ Preprocess Tweets", use_container_width=True)
    if preprocess_btn:
        st.session_state.pipeline_status = "preprocessing"
        try:
            import pandas as pd
            from datasets import Dataset
            from transformers import GPT2Tokenizer
            # Save fetched tweets to CSV
            df = pd.DataFrame(st.session_state.fetched_tweets)
            csv_path = Path("data/tweets_data.csv")
            df.to_csv(csv_path, index=False)
            # Clean and tokenize
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            df["tweet_text"] = df["tweet_text"].astype(str)
            # Remove noise
            import re
            def remove_noise(text):
                text = re.sub(r"http\S+|www\.\S+", "", text)
                text = re.sub(r"@\w+", "", text)
                text = re.sub(r"#(\w+)", r"\1", text)
                text = re.sub(r"\s+", " ", text)
                return text.strip()
            df["tweet_text"] = df["tweet_text"].map(remove_noise)
            # Tokenize
            separator = "<|endoftext|>"
            df["tweet_text"] = df["tweet_text"] + separator
            dataset = Dataset.from_pandas(df)
            def tokenize_function(examples):
                return tokenizer(examples["tweet_text"], truncation=True, max_length=280)
            tokenized = dataset.map(tokenize_function, batched=True)
            # Split
            split = tokenized.train_test_split(test_size=0.2, seed=42)
            out_dir = Path("data/processed")
            out_dir.mkdir(parents=True, exist_ok=True)
            split["train"].save_to_disk(out_dir / "train_data")
            split["test"].save_to_disk(out_dir / "test_data")
            tokenizer.save_pretrained(out_dir / "tokenizer")
            st.session_state.preprocessed = True
            st.success("✅ Preprocessing complete!")
        except Exception as exc:
            st.error(f"❌ Preprocessing failed: {str(exc)}")
        st.session_state.pipeline_status = "idle"

    # Step 3: Train Model
    train_btn = st.button("3️⃣ Train Model", use_container_width=True)
    if train_btn:
        st.session_state.pipeline_status = "training"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
            from datasets import load_from_disk
            train_path = Path("data/processed/train_data")
            test_path = Path("data/processed/test_data")
            tokenizer_path = Path("data/processed/tokenizer")
            model_name = "distilgpt2"
            output_dir = Path("model/fine_tuned_model")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            train_dataset = load_from_disk(str(train_path))
            test_dataset = load_from_disk(str(test_path))
            model = AutoModelForCausalLM.from_pretrained(model_name)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=2,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=50,
                learning_rate=2e-5,
                weight_decay=0.01,
                fp16=False,
                report_to=[],
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            with st.spinner("🚀 Training model (2 epochs)..."):
                trainer.train()
                trainer.save_model(str(output_dir))
                tokenizer.save_pretrained(str(output_dir))
            st.session_state.trained = True
            st.success("✅ Model training complete!")
        except Exception as exc:
            st.error(f"❌ Training failed: {str(exc)}")
        st.session_state.pipeline_status = "idle"

    # Step 4: Generate Tweets
    st.divider()
    st.subheader("✍️ Generate Tweets")
    prompt = st.text_input(
        "💡 Enter a tweet prompt:",
        placeholder="e.g., 'AI is revolutionizing...' or 'Just finished...'",
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        num_tweets = st.slider("Number of tweets to generate", 1, 5, 3)
    with col2:
        style = st.selectbox("Style", list(STYLE_TEMPS.keys()))
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_new_tokens = st.slider("Max new tokens", 20, 120, 80, 5)
            top_k = st.slider("Top-k", 10, 200, 50, 5)
        with col2:
            top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.95, 0.01)
            repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
        no_repeat_ngram_size = st.slider("No-repeat n-gram size", 1, 4, 2)
        seed = st.text_input("Optional seed (integer)", value="", placeholder="Leave empty for random")
    gen_btn = st.button("4️⃣ Generate Tweets", use_container_width=True)
    if gen_btn:
        prompt_clean = prompt.strip()
        if len(prompt_clean) < 3:
            st.error("❌ Please enter a prompt with at least 3 characters.")
        elif not st.session_state.trained:
            st.error("❌ Model not trained yet. Run training step above.")
        else:
            if seed.strip():
                try:
                    set_seed(int(seed.strip()))
                except ValueError:
                    st.warning("⚠️ Seed must be an integer. Using random seed.")
            try:
                generator = load_generator()
                with st.spinner("✨ Generating tweets..."):
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
                st.success(f"✅ Generated {num_tweets} tweets!")
                for idx, tweet in enumerate(tweets, start=1):
                    st.markdown(f"**Tweet {idx}** ({style}):\n> {tweet}")
            except Exception as exc:
                st.error(f"❌ Generation failed: {str(exc)}")
    
    # Save & History Section
    if st.session_state.latest:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📌 Save to History", use_container_width=True):
                st.session_state.history.extend(st.session_state.latest)
                st.success(f"✅ Saved {len(st.session_state.latest)} tweets to history!")
        with col2:
            if st.button("🗑️ Clear Generated Tweets", use_container_width=True):
                st.session_state.latest = []
                st.rerun()
    
    # Tweet History
    st.subheader("📚 Tweet History")
    
    if st.session_state.history:
        st.write(f"Total saved tweets: **{len(st.session_state.history)}**")
        
        with st.expander(f"View all {len(st.session_state.history)} tweets", expanded=True):
            for idx, tweet in enumerate(st.session_state.history, start=1):
                st.write(f"{idx}. {tweet}")
        
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history.clear()
            st.success("✅ History cleared.")
            st.rerun()
    else:
        st.info("📭 No tweets in history yet. Generate and save some!")


if __name__ == "__main__":
    main()
