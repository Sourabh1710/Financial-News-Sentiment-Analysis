# Financial News Sentiment + Stock Movement

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FinBERT](https://img.shields.io/badge/FinBERT-ProsusAI-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-75.1%25-success?style=flat)
![Correlation](https://img.shields.io/badge/Correlation-Methodology%20Demo-lightgrey?style=flat)

> **Does today's financial news predict tomorrow's stock move?**  
> An NLP pipeline that classifies financial headlines with FinBERT (75.1% accuracy, zero-shot) and builds a complete sentiment-price correlation pipeline — including an honest negative result that exposes a real data limitation rather than overclaiming a trading signal.

[Live Demo →](https://your-app.streamlit.app) &nbsp;|&nbsp; [Key Findings ↓](#-what-the-model-found)

---

## The Business Problem

News sentiment is a live trading signal. Bloomberg Terminal charges ~$24,000/year per seat partly for its sentiment analytics layer. RavenPack — which processes financial news into structured sentiment feeds for hedge funds — was acquired in 2023 at a reported $600M valuation.

The challenge is that financial language is a dialect, not just English. "Downgrade", "coupon", "spread", "rally", "guidance" mean specific things in financial context that no general-purpose NLP tool understands. VADER, the standard rule-based sentiment tool, scores only 37.6% on financial headlines — worse than random on a 3-class problem.

This project builds the core pipeline from scratch: classify financial headlines with a domain-pretrained BERT model, aggregate daily signals, and test whether they predict the next trading session — while being explicit about what the result does and doesn't prove.

![Label Distribution](outputs/eda_sentiment_distribution.png)
*Label distribution across 5,842 Financial PhraseBank sentences. Negative headlines (14.7%) are the minority class — and the hardest to get right.*

---

## Results

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Notes |
|---|---|---|---|---|---|
| VADER (rule-based) | 37.6% | 0.12 | 0.46 | 0.36 | No financial domain knowledge |
| TF-IDF + Logistic Regression | 66.4% | 0.40 | 0.74 | 0.69 | Bag-of-words, ~30s to train |
| TF-IDF + SVM | 67.2% | 0.14 | 0.75 | 0.71 | Best classical model |
| **FinBERT (ProsusAI)** | **75.1%** | **0.60** | **0.78** | **0.79** | **Zero-shot — no fine-tuning** |

**Test set:** 1,169 held-out sentences, stratified 80/20 split, all four models evaluated on the identical split.

**Sentiment → next-day return correlation:** Computed across AAPL, MSFT, GOOGL, AMZN, TSLA over 500 trading days. **Result: no statistically significant correlation for any ticker** (all p-values > 0.3, direction accuracy 50.0–50.8%). See [Why the correlation result is null](#why-the-correlation-result-is-null) — this is expected given a real limitation in the dataset, not a failed analysis.

> **On the negative F1 column:** Headline accuracy is a misleading metric here. SVM scores 67.2% overall but achieves negative F1 of only 0.14 — it almost never correctly identifies bearish headlines. FinBERT, used **zero-shot** (no task-specific fine-tuning), more than quadruples this to 0.60. Since early detection of negative signals is the highest-value use case for a trading desk, this is the comparison that matters — and it holds even before any fine-tuning.

![Model Comparison](outputs/model_comparison.png)
*Left: accuracy by model with majority-class baseline (53.6%). Right: per-class F1 — FinBERT's gain on the negative class is the core classification result.*

---

## What the Model Found

**1. Generic NLP is blind to financial language (VADER: 37.6%)**  
VADER was built on social media and general English. It has no concept of "downgrade" as bearish or "raised guidance" as bullish. Its 37.6% accuracy on financial headlines is worse than a coin flip on a 3-class problem — the motivation for every other design decision in this project.

**2. Bag-of-words models plateau around 67% — an architectural ceiling, not a tuning problem**  
TF-IDF treats each word independently. "The company did not profit" and "the company recorded a profit" both contain the token "profit" and score identically. Logistic Regression and SVM both land around 67% regardless of hyperparameter tuning, because neither has a mechanism for understanding word order, negation, or context.

**3. Domain-pretrained BERT improves accuracy by 8 points — zero-shot, with no training on this data**  
FinBERT (ProsusAI) was pretrained on 4.9 billion words of Reuters, Bloomberg, and earnings call transcripts, then fine-tuned for sentiment by its authors — but **not on this exact dataset**. Used directly, with zero fine-tuning, it improves overall accuracy from 67.2% to 75.1% and roughly quadruples negative-class F1 (0.14 → 0.60). The commonly-cited ~91-97% FinBERT figures come from evaluating on the *AllAgree* subset of Financial PhraseBank (sentences where all 16 annotators agreed) — the easiest examples. This dataset includes the full agreement spectrum, including genuinely ambiguous sentences that even trained financial analysts disagreed on.

**4. One near-tie misclassification reveals model uncertainty, not model failure**  
"Analysts upgraded the stock citing bullish revenue growth" (true label: positive) was predicted **negative at 49.2% vs. positive at 48.3%** — essentially a coin flip. This is a useful finding: the model isn't confidently wrong, it's genuinely uncertain on a sentence containing both "upgraded"/"bullish" (positive signals) in a syntactic frame ("Analysts upgraded... citing...") that may pattern-match toward analyst-action language the model associates with caution. A production system would flag predictions below ~60% confidence for human review rather than act on them automatically.

### Why the correlation result is null

Across all five tickers, sentiment-vs-next-day-return correlations were negligible and non-significant (Pearson r between -0.05 and +0.01, all p > 0.3), with direction accuracy of 50.0-50.8% — statistically identical to a coin flip.

**This is the expected outcome, not a failed experiment.** Financial PhraseBank sentences carry no publication timestamps. To run the correlation pipeline at all, the 5,842 headlines were spread evenly across 503 trading days — an arbitrary alignment with zero temporal meaning. There is no real signal in the data for the correlation to find, so finding none is exactly correct.

What this section of the project demonstrates is the **pipeline**, not a market finding: daily sentiment aggregation, alignment with next-day returns, and Pearson/Spearman/direction-accuracy testing with proper p-value reporting. To turn this into a real signal, the only missing piece is a news source with **actual publication timestamps** — a financial news API (e.g. Alpha Vantage News Sentiment, NewsAPI, or a scraped RSS feed with dates) — so each headline maps to the trading day it was actually published. The rest of the pipeline plugs in unchanged.

> Reporting "no significant correlation, here's exactly why, here's what's needed for a real test" is a stronger signal of analytical maturity than reporting a spurious significant result and not questioning it — overclaiming false correlations is one of the fastest ways to lose credibility on a quant or risk team.

---

## Technical Approach

### Why these specific choices

**FinBERT over generic BERT** — Generic BERT (Wikipedia + BooksCorpus) typically scores in the high 70s-low 80s on Financial PhraseBank. FinBERT's domain pretraining on financial corpora gives it a real, measurable edge on financial vocabulary and negation — even zero-shot, as shown above.

**Spearman over Pearson correlation** — Financial returns have fat tails. A single ±10% day would dominate a Pearson calculation. Spearman's rank-based correlation is robust to these extreme events, which is why it's preferred in quantitative finance for non-normal return series. (Both are reported here — both agree the correlation is null.)

**Next-day return, not same-day** — Correlating sentiment with same-day returns conflates cause and effect: news published at 11am may react to a price move that happened at 9:30am open. Next-day return measures whether today's information has forward-looking content — the correct causal direction for an actionable signal.

**Raw text for FinBERT, preprocessed for classical models** — BERT's WordPiece tokeniser handles subword segmentation, capitalisation, and punctuation as meaningful signals; preprocessing before BERT would destroy this context. For TF-IDF, preprocessing is essential — without it, "profit", "Profit", "PROFIT" and "profits" are four different vocabulary tokens, wasting feature space.

**Calibrated SVM for probability outputs** — `LinearSVC` doesn't natively output probabilities. Wrapping it in `CalibratedClassifierCV` applies isotonic calibration so the model produces confidence scores comparable to Logistic Regression's `predict_proba` — used for the per-class confidence display in the dashboard.

### Domain-aware preprocessing decisions

| Decision | What it does | Why it matters |
|---|---|---|
| Preserve financial signal words | "bullish", "bearish", "downgrade", "rally" excluded from stop-word removal | Generic stop-word lists would delete the most informative tokens in financial text |
| Protect negation tokens | "no", "not", "nor" retained despite being stop words | "no profit" and "profit" are antonyms; removing "no" collapses the distinction |
| Keep `$` and `%` characters | Removed from punctuation strip list | "$2.5B write-down" and "fell 8%" carry magnitude signal |
| Bigram TF-IDF (ngram_range 1-2) | Captures "profit growth", "revenue decline" as features | Two-word financial phrases carry more signal than either word individually |
| Class-weighted training | `class_weight='balanced'` on all classical models | 14.7% negative class causes models to ignore bearish signals without compensation |

---

## Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, numpy |
| NLP preprocessing | NLTK (tokenisation, lemmatisation, stop words) |
| Classical models | scikit-learn (TF-IDF, Logistic Regression, LinearSVC, Pipeline) |
| Rule-based baseline | VADER (vaderSentiment) |
| Deep learning | HuggingFace Transformers, PyTorch, ProsusAI/finbert |
| Finance data | yfinance |
| Statistics | scipy (Pearson, Spearman, p-values) |
| Visualisation & App | Plotly, Streamlit |

---

## Quickstart

```bash
git clone https://github.com/yourusername/financial-sentiment
cd financial-sentiment

pip install -r requirements.txt

# Train and save classical models (~30 seconds)
python src/classical_models.py

# Score all headlines with FinBERT and run correlation analysis
# (~1hr on CPU — writes outputs/scored_headlines.csv, daily_sentiment.csv, stock_data.csv)
python src/correlation.py

# Launch dashboard — loads precomputed outputs/*.csv instantly
streamlit run app/app.py
```

The app opens at `localhost:8501`. Tab 1 lets you paste any financial headline for a real-time FinBERT prediction with per-class confidence. Tab 3 shows the full sentiment-vs-price analysis with the methodology caveat above displayed inline.

---

## Project Structure

```
├── src/
│   ├── data_loader.py              # Dataset loading, EDA, class balance analysis
│   ├── preprocessor.py             # Finance-aware NLP pipeline (domain vocab preservation)
│   ├── classical_models.py         # TF-IDF baselines with cross-validation and calibration
│   ├── finbert_model.py            # FinBERT inference wrapper with offline fallback
│   ├── stock_data.py               # yfinance OHLCV fetcher + GBM simulation fallback
│   ├── correlation.py              # Daily sentiment aggregation + Spearman correlation
│   └── compute_finbert_metrics.py  # Recomputes FinBERT metrics from scored_headlines.csv
├── app/
│   └── app.py                      # Four-tab Streamlit dashboard
├── data/
│   └── financial_phrasebank.csv    # Source dataset (5,842 labelled headlines)
├── models/                         # Saved sklearn pipelines (generated by classical_models.py)
├── outputs/                        # Precomputed scored_headlines.csv, daily_sentiment.csv, stock_data.csv
└── requirements.txt
```

---

## Possible next steps

- **Fine-tune FinBERT** on the 80% training split (the original target was 90%+ accuracy). Zero-shot already gets 75.1% — fine-tuning on this dataset's specific label distribution would likely close most of the gap to the AllAgree-subset benchmarks.
- **Real-dated headlines** via a financial news API would let the correlation pipeline (already built and tested) produce a meaningful result instead of a methodology demo.
- **Confidence-based filtering**: only act on predictions above ~70% confidence, given the near-tie example found in Tab 1.

---

*Dataset: [Financial PhraseBank — Malo et al., 2014](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) · 5,842 sentences · annotated by 16 finance professionals*
