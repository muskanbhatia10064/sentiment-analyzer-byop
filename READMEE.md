# 💬 Sentiment Analyzer using NLP & Machine Learning

> BYOP Project | CSA2001 – Fundamentals in AI and ML | VIT Bhopal

---

##  Problem Statement

Millions of product reviews are written every day. Reading them manually is impossible. This project builds an NLP-powered classifier that automatically detects whether a review is **Positive**, **Negative**, or **Neutral** — enabling automated opinion mining at scale.

---

## What This Project Does

Given a product review in plain English, the system:
1. **Preprocesses** the text (lowercase, remove punctuation, remove stopwords)
2. **Vectorizes** it using TF-IDF (Term Frequency-Inverse Document Frequency)
3. **Classifies** it into one of three sentiment categories

| Sentiment | Example Review |
|-----------|----------------|
| Positive | "Amazing product, exceeded all my expectations!" |
|  Negative | "Terrible quality, broke after just two days." |
|  Neutral  | "It works okay, nothing special but does the job." |

---

##  Models Trained & Compared

| Model | Test Accuracy | CV Score (5-Fold) |
|-------|:---:|:---:|
| **Naive Bayes**  | **87.50%** | 88.63% ± 6.70% |
| Logistic Regression | 87.50% | 86.53% ± 9.02% |
| Linear SVM | 87.50% | 87.63% ± 9.40% |

---

## 📁 Project Structure

```
sentiment_analyzer/
│
├── sentiment_analyzer.py         # Main NLP + ML pipeline
├── sentiment_dataset.csv         # Dataset (120 labeled reviews)
├── eda_plots.png                 # EDA visualizations
├── word_frequency.png            # Top words per sentiment class
├── model_results.png             # Accuracy charts + confusion matrices
├── Project_Report.docx           # Full project report
└── README.md                     # This file
```

---

##  Setup & Installation

### Prerequisites
- Python 3.8 or above
- pip

### Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the project

```bash
python sentiment_analyzer.py
```

The script will automatically:
1. Build the dataset
2. Perform EDA and save plots
3. Preprocess all text
4. Train all 3 models
5. Print accuracy and classification reports
6. Save visualization files
7. Run live predictions on 6 test sentences

---

##  How It Works (Quick Summary)

```
Raw Text Review
      ↓
Preprocessing (lowercase → remove punctuation → remove stopwords)
      ↓
TF-IDF Vectorization (text → numbers)
      ↓
Train/Test Split (80% train, 20% test)
      ↓
Train 3 Models (Naive Bayes, Logistic Regression, Linear SVM)
      ↓
Cross-Validate (5-fold)
      ↓
Evaluate (Accuracy, F1-Score, Confusion Matrix)
      ↓
Predict New Reviews 
```

---

##  Live Prediction Example

```python
# Input
review = "This is the best product I have ever bought, absolutely love it!"

# Output
# [Positive] — Predicted correctly!
```

---

##  Course Concepts Applied (CSA2001)

| Concept | Course Outcome |
|---------|---------------|
| NLP & Sentiment Analysis | CO5 |
| Supervised Classification | CO4 |
| Probability Theory (Naive Bayes) | CO3 |
| TF-IDF Feature Learning | CO3 |
| Cross-Validation & Overfitting | CO4 |
| Bias-Variance Tradeoff | CO4 |

---

##  Future Improvements

- Use BERT or LSTM for deeper contextual understanding
- Connect to Twitter/Amazon API for real-time review analysis
- Add aspect-based sentiment (price, quality, delivery separately)
- Build a Flask web app for live predictions

---

##  References

1. S. Russell & P. Norvig — *Artificial Intelligence: A Modern Approach*, 3rd Ed.
2. Ethem Alpaydin — *Machine Learning: The New AI*, MIT Press
3. [Scikit-learn Documentation](https://scikit-learn.org)
4. Pang, B. & Lee, L. — *Opinion Mining and Sentiment Analysis*, 2008

---

## Author

**MUSKAN BHATIA**
SCAI
VIT Bhopal University
Academic Year 2025–2026
