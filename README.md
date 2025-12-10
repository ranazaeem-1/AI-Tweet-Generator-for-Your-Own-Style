# 🐦✨ AI Tweet Generator for Your Own Style

> **Create tweets in your unique style using GPT-2, all from a beautiful Streamlit app!**

---

## 🚀 Features

- 🔐 Secure credential management (.env support)
- 📥 Fetch tweets from your timeline (X API)
- 🧹 Preprocess and clean tweets
- 🏋️‍♂️ Fine-tune GPT-2 on your data
- ✍️ Generate tweets with custom prompts
- 🎛️ Advanced generation controls
- 📚 Tweet history and export
- ⚠️ Free tier API warnings
- 🖥️ All-in-one Streamlit UI (no scripts needed!)

---

## 📦 Project Structure

```
AI-Tweet-Generator-for-Your-Own-Style/
├── app/
│   └── app.py            # Streamlit web app (run everything here!)
├── src/                  # Legacy CLI scripts (not needed for UI)
│   ├── fetch_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── generate_tweet.py
├── model/
│   └── fine_tuned_model/ # Your trained model
├── data/
│   └── tweets_data.csv   # Fetched tweets
├── requirements.txt      # Dependencies
├── .env.example          # Credential template
└── README_NEW.md         # This file
```

---

## 🌈 Quick Start (Streamlit UI)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up your credentials**
   - Copy `.env.example` to `.env`
   - Paste your X API keys in `.env`
3. **Launch the app**
   ```bash
   streamlit run app/app.py
   ```
4. **Use the UI!**
   - Enter your credentials (auto-loaded from `.env`)
   - Enter your Twitter handle
   - Click through: **Fetch → Preprocess → Train → Generate**
   - Enjoy your personalized AI tweets!

---

## 🖥️ Streamlit App Workflow

| Step         | What Happens?                  |
|--------------|-------------------------------|
| 1️⃣ Fetch    | Downloads your tweets         |
| 2️⃣ Preprocess | Cleans & tokenizes tweets     |
| 3️⃣ Train    | Fine-tunes GPT-2 on your data |
| 4️⃣ Generate | Creates tweets in your style  |

**No manual script running required!**

---

## ⚠️ X API Free Tier Limitations

| Limit                | Value      | Impact                |
|----------------------|------------|-----------------------|
| Tweets to read/month | 100        | Limited training data |
| Tweets to post/month | 50         | Limited testing       |
| Timeline access      | v1.1 only  | Some endpoints blocked|

**If you see 403/453 errors:**
- You are on the Free tier. Upgrade to Basic+ for full timeline access: [X Pricing](https://developer.x.com/en/pricing)
- You can only fetch tweets from your own account, not others.

---

## 💡 Tips for Best Results

- **Use 50+ tweets** for good style capture
- **Clean your data** (remove URLs/mentions)
- **Adjust temperature** for creativity
- **Set a seed** for reproducible results
- **Train on recent tweets** for current style

---

## 🛠️ Troubleshooting

- **Missing environment variable:**
  - Set your API keys in `.env` or sidebar
- **403/453 Forbidden:**
  - Upgrade your X API tier
- **Model not found:**
  - Run the training step in the app
- **CUDA out of memory:**
  - Lower batch size or use CPU

---

## 📚 FAQ

**Q: Can I use this with other LLMs?**  
A: Yes! Swap `distilgpt2` for any HuggingFace model.

**Q: How many tweets do I need?**  
A: 20-30 minimum, 50-100 ideal.

**Q: Can I deploy this online?**  
A: Yes! Streamlit Cloud, Hugging Face Spaces, etc.

**Q: Does this work offline?**  
A: Yes! Only fetching tweets needs internet.

---

## 👤 Author

Developed by **Azam Effendi** (@effendii69)

---

## 🏁 Next Steps

1. 🚀 Train your model in the app
2. ✨ Generate tweets
3. 📤 Share your results
4. 🛠️ Tune parameters for better style
5. 🌍 Deploy for friends to use

---

## 🛡️ Security

- Never commit `.env` or API keys
- Add `data/*.csv` and `model/*` to `.gitignore`
- Rotate credentials if compromised

---

## 📢 Enjoy your personalized AI Tweet Generator! 🐦✨
