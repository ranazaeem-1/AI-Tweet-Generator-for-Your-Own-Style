import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import tweepy

BASE_DIR = Path(__file__).resolve().parents[1]
MIN_REQUEST = 1
MAX_FREE_TWEETS = 50  # Free tier cap per month for timeline reads


def _get_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing environment variable: {key}")
    return value


def authenticate_from_args(args: argparse.Namespace) -> tweepy.API:
    """
    Build an authenticated Tweepy API client using CLI args or environment variables.
    Env fallbacks: TWITTER_API_KEY, TWITTER_API_KEY_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET.
    """
    api_key = args.api_key or _get_env("TWITTER_API_KEY")
    api_key_secret = args.api_key_secret or _get_env("TWITTER_API_KEY_SECRET")
    access_token = args.access_token or _get_env("TWITTER_ACCESS_TOKEN")
    access_token_secret = args.access_token_secret or _get_env("TWITTER_ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
    return tweepy.API(auth, wait_on_rate_limit=True)


def fetch_user_tweets(
    api: tweepy.API,
    screen_name: str,
    max_items: int = 100,
    include_rts: bool = False,
    exclude_replies: bool = True,
) -> List[dict]:
    target_items = max(min(max_items, MAX_FREE_TWEETS), MIN_REQUEST)
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
    except tweepy.TweepyException as exc:
        code = getattr(exc, "api_code", None)
        if code == 88:
            raise RuntimeError("Rate limit exceeded. Free tier allows ~100 posts/month; try again later.") from exc
        if code in (403, 453):
            raise RuntimeError(
                "Access forbidden for timeline reads. Free tier limits apply; Basic/Pro/Elevated access may be required."
            ) from exc
        raise
    return tweets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tweets from a user timeline (free-tier safe defaults capped at 100/month)."
    )
    parser.add_argument("--user", required=True, help="Twitter handle without @ (screen name).")
    parser.add_argument("--output", default=str(BASE_DIR / "data" / "tweets_data.csv"), help="Path to write the CSV file.")
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_FREE_TWEETS,
        help=f"Maximum number of tweets to fetch (capped at free-tier limit {MAX_FREE_TWEETS}).",
    )
    parser.add_argument("--include-rts", action="store_true", help="Include retweets in the export.")
    parser.add_argument("--include-replies", action="store_true", help="Include replies (off by default).")
    parser.add_argument("--api-key", help="Twitter API key (fallback TWITTER_API_KEY).")
    parser.add_argument("--api-key-secret", help="Twitter API key secret (fallback TWITTER_API_KEY_SECRET).")
    parser.add_argument("--access-token", help="Twitter access token (fallback TWITTER_ACCESS_TOKEN).")
    parser.add_argument("--access-token-secret", help="Twitter access token secret (fallback TWITTER_ACCESS_TOKEN_SECRET).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = authenticate_from_args(args)

    tweets = fetch_user_tweets(
        api=api,
        screen_name=args.user,
        max_items=args.max,
        include_rts=args.include_rts,
        exclude_replies=not args.include_replies,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(tweets, columns=["tweet_text", "date", "id"])
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} tweets to {output_path}")


if __name__ == "__main__":
    main()
