# Sarcasm Detection in News Headlines

Detects sarcastic news headlines with a custom NLP preprocessing pipeline and Naive Bayes classifiers, largely built **from scratch** (tokenisation, bag-of-words features and Naive Bayes) to show how each step works.

**Tools:** Python · pandas · NumPy · regex · scikit-learn (evaluation and baselines)

## Dataset
[News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) (Kaggle): headlines from *The Onion* (sarcastic) and *HuffPost* (not sarcastic).
Download `Sarcasm_Headlines_Dataset_v2.json` into `data/`.

## What's inside
| Task | Contents |
|---|---|
| 1. Preprocessing | Lowercasing, HTML stripping, **sarcasm marker tokens** (`!!` → `[EXC_SEQ]`, `?!` → `[INT_MARK]`, `...` → `[ELLIP]`), negation handling, bigrams |
| 2. Features | Hand-built bag-of-words and binary feature representations |
| 3. Naive Bayes | **Multinomial vs Bernoulli** NB with Laplace smoothing tuned on a validation set |
| 4. Numeric features | Engineered counts (exclamation marks, digits, length…) with imputation |
| 5. Outliers | Detection and treatment of extreme numeric features |
| 6. Scaling | MinMax vs standardisation for Gaussian NB |
| 7. Class imbalance | Random oversampling, judged on F1 as well as accuracy |
| 8. Error analysis | Inspection of misclassified headlines |

## Results
- **Multinomial NB: ~81% validation accuracy**, compared with ~66% for Bernoulli NB. Word and bigram frequency carries real signal.
- Numeric features alone performed at chance (~50%) with Gaussian NB, which shows the signal is in the language, not in surface statistics.
- Oversampling improved recall on the sarcastic class. F1 is the more honest metric here.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook sarcasm_detection.ipynb
```

---
*MSc Data Science & Analytics, Munster Technological University (Applied Machine Learning)*
