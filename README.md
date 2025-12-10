<img width="1150" height="815" alt="image" src="https://github.com/user-attachments/assets/73ee86ee-50e3-4274-b35f-455e007b83c3" />

AI Tweet Generator
===================

Generate tweets in your style by fine-tuning GPT-2 on your own timeline data.

Project structure
-----------------
```
AI_Tweet_Generator/
├── data/                    # Raw and processed data
│   └── tweets_data.csv      # Exported tweets (created by fetch_data.py)
├── model/
│   └── fine_tuned_model/    # Saved fine-tuned model artifacts
│       └── pytorch_model.bin
├── src/                     # Core scripts
│   ├── fetch_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── generate_tweet.py
├── app/
│   └── app.py               # Streamlit UI
├── requirements.txt
└── README.md
```

Setup
-----
1) Install dependencies (ideally in a virtualenv):
```
pip install -r requirements.txt
```

2) Set Twitter API credentials (OAuth 1.0a):
```
set TWITTER_API_KEY=...
set TWITTER_API_KEY_SECRET=...
set TWITTER_ACCESS_TOKEN=...
set TWITTER_ACCESS_TOKEN_SECRET=...
```

Usage
-----
1) Fetch up to 3,000 tweets to CSV:
```
python src/fetch_data.py --user your_handle --output data/tweets_data.csv --max 3000
```
Flags: `--include-rts` to include retweets, `--include-replies` to keep replies, or pass keys via CLI (`--api-key`, etc.).

2) Preprocess/tokenize (optional cleaning of URLs/mentions/hashtags):
```
python src/preprocess_data.py --input data/tweets_data.csv --output-dir data/processed --clean
```

3) Fine-tune distilgpt2:
```
python src/train_model.py --train-data data/processed/train_data --test-data data/processed/test_data --output-dir model/fine_tuned_model
```
Adjust `--epochs`, `--train-batch-size`, `--learning-rate` as needed; add `--fp16` if supported.

4) Generate tweets from the fine-tuned model:
```
python src/generate_tweet.py --prompt "Today I learned" --model-path model/fine_tuned_model --num-tweets 3
```

5) Streamlit UI:
```
streamlit run app/app.py
```

Notes
-----
- Defaults use paths relative to the repository root even when run from inside `src/`.
- Ensure GPU/torch setup for faster training; CPU works for small experiments but will be slow.


developed by azameffendi 
