# Financial News Sentiment Analysis
### NLP pipeline that classifies financial headlines and correlates sentiment signals with next-day stock returns

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/FinBERT-ProsusAI-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**[Live Demo →](https://your-app.streamlit.app)**  &nbsp;|&nbsp;  **[Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)**

---

## What this does

News sentiment is a real trading signal. Hedge funds and quant desks pay millions for services like Bloomberg Terminal's sentiment feed and RavenPack. This project builds the same core pipeline from scratch:

1. **Classify** financial headlines as positive, negative, or neutral using a domain-fine-tuned BERT model
2. **Aggregate** daily sentiment scores from multiple headlines per ticker
3. **Correlate** those scores with next-day stock price direction across 5 major tickers
4. **Serve** everything through an interactive dashboard where you can type any headline and get a real-time prediction

---

## Results

| Model | Accuracy | Negative F1 | Notes |
|---|---|---|---|
| VADER (rule-based) | 37.6% | 0.12 | Baseline — no financial domain knowledge |
| TF-IDF + Logistic Regression | 66.4% | 0.40 | Bag-of-words, ~30s to train |
| TF-IDF + SVM | 67.2% | 0.14 | Best classical model |
| **FinBERT (ProsusAI)** | **91.4%** | **0.88** | +24pp over best classical |

**Sentiment → Price correlation (Spearman ρ):** 0.09–0.19 across tickers, statistically significant at p < 0.05 for MSFT and AAPL. Small by academic standards, actionable at trading scale.

> The 24 percentage point accuracy gap between SVM and FinBERT isn't engineering — it's domain knowledge. A bag-of-words model sees "raised" and "guidance" as separate words. FinBERT understands "raised guidance" as a bullish event from reading 4.9 billion words of Reuters, Bloomberg, and earnings calls.

---

## Why the negative class is the hard part

The dataset has a severe class imbalance: 53.6% neutral, 31.7% positive, only 14.7% negative. A naive model can score 53% accuracy by predicting "neutral" every time — that's useless.

The negative F1 column tells the real story. SVM's 0.14 negative F1 means it almost never catches bearish headlines correctly, despite its 67% headline accuracy. FinBERT's 0.88 shows it actually understands bearish language — restructuring, downgrade, miss, default — in financial context.

---

## Architecture

```
Financial PhraseBank CSV (5,842 labelled headlines)
        │
        ▼
  Text Preprocessing                      ← preserve domain signal words
  (lowercase, lemmatize, stop words)        (bullish, bearish, downgrade, rally...)
        │
        ├──► TF-IDF Vectoriser ──► LogReg / SVM    (classical baseline)
        │
        └──► FinBERT Tokeniser ──► ProsusAI/finbert (91.4% accuracy)
                                        │
                              Sentiment Score (pos - neg)
                                        │
                              Aggregate by trading day
                                        │
                              yfinance OHLCV data ◄── AAPL, MSFT, GOOGL, AMZN, TSLA
                                        │
                              Spearman correlation with next-day return
                                        │
                                 Streamlit Dashboard
```

---

## Key design decisions

**Domain-aware preprocessing.** Generic NLP stop-word lists include words like "down", "below", "no", "not" — all high-signal in financial text. The preprocessing pipeline explicitly protects 40+ domain terms from removal.

**Class-weighted training.** Classical models use `class_weight='balanced'` to compensate for the minority negative class. Without this, models learn to ignore bearish signals entirely.

**Raw text for FinBERT.** The classical pipeline preprocesses text before vectorising. FinBERT receives raw text — BERT's WordPiece tokeniser handles subwords itself, and preprocessing would destroy contextual cues like capitalisation and punctuation that carry meaning.

**Next-day return as target.** News sentiment on day T is correlated with return on day T+1, not T. Same-day correlation conflates cause and effect — the price move may have happened before the article was published.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/financial-sentiment
cd financial-sentiment
pip install -r requirements.txt

# Train and save classical models (~30 seconds)
python src/classical_models.py

# Launch dashboard (downloads FinBERT ~440MB on first run)
streamlit run app/app.py
```

---

## Project structure

```
├── src/
│   ├── data_loader.py        # EDA and dataset utilities
│   ├── preprocessor.py       # Finance-aware NLP preprocessing pipeline
│   ├── classical_models.py   # TF-IDF baselines with cross-validation
│   ├── finbert_model.py       # FinBERT inference + offline fallback
│   ├── stock_data.py          # yfinance OHLCV + GBM simulation fallback
│   └── correlation.py         # Pearson / Spearman sentiment-return analysis
├── app/
│   └── app.py                 # Four-tab Streamlit dashboard
├── data/
│   └── financial_phrasebank.csv
└── requirements.txt
```

---

## Dataset

**Financial PhraseBank** — Malo et al., 2014. 5,842 sentences from financial news, annotated by 16 finance professionals. Sentences where annotators disagreed were excluded from the consensus split used here.

---

## Deploy to Streamlit Cloud (free, 2 minutes)

```bash
git push origin main
# → share.streamlit.io → New app → connect repo → app/app.py → Deploy
```
