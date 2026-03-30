"""
Sentiment Analyzer using NLP & Machine Learning
CSA2001 - Fundamentals in AI and ML | BYOP Project
VIT Bhopal

Classifies text reviews as Positive, Negative, or Neutral
using TF-IDF vectorization and multiple ML classifiers.
"""

#Add main sentiment analyzer code"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET
# ─────────────────────────────────────────────

POSITIVE_REVIEWS = [
    "This product is absolutely amazing and works perfectly",
    "I love this! Best purchase I have ever made",
    "Excellent quality and fast delivery, very satisfied",
    "Outstanding performance, highly recommend to everyone",
    "Great value for money, very happy with this",
    "Fantastic product, exceeded all my expectations",
    "Really good quality, will definitely buy again",
    "Super impressed with how well this works",
    "Brilliant product, exactly what I was looking for",
    "Wonderful experience, the product is top notch",
    "Very pleased with this purchase, works like a charm",
    "Incredible product, simply the best on the market",
    "Happy with the purchase, good quality and durable",
    "Five stars! Absolutely love this product",
    "Perfect in every way, very high quality build",
    "Amazing value, could not be happier with this",
    "Superb quality, delivered on time and well packaged",
    "Best product I have used in years, highly satisfied",
    "Loved it so much, bought one for my friend too",
    "Excellent product, works exactly as described",
    "Very good, quick shipping and well packaged product",
    "Impressed by the quality, strongly recommend this",
    "Works great, very easy to use and reliable",
    "So happy with this, it makes my life easier",
    "Top quality product, no complaints whatsoever",
    "Awesome product, fast delivery and great packaging",
    "Incredible build quality and very easy to set up",
    "Really satisfied with this purchase, great item",
    "Good product, performed exactly as I expected",
    "Loved the product, great customer service too",
    "This is really nice, I am very impressed overall",
    "Highly recommend this to anyone looking for quality",
    "Perfect product at a perfect price, very happy",
    "Excellent! Delivered early and works brilliantly",
    "Outstanding value, much better than I expected",
    "Great product, well made and easy to use daily",
    "Very happy customer, will order again for sure",
    "Absolutely wonderful, surpassed my expectations completely",
    "Quality is top notch and delivery was super fast",
    "Brilliant purchase, everything about it is great",
]

NEGATIVE_REVIEWS = [
    "Terrible product, broke after just two days of use",
    "Very disappointed, this is not worth the money at all",
    "Awful quality, completely useless and badly made",
    "Worst purchase I ever made, total waste of money",
    "Do not buy this, it stopped working immediately",
    "Extremely poor quality, fell apart after one week",
    "Very unhappy with this, nothing works as described",
    "Horrible product, arrived damaged and unusable",
    "Complete rubbish, does not work at all as advertised",
    "Terrible experience, product is cheap and flimsy",
    "Garbage product, broke on the very first day",
    "Absolutely dreadful, would not recommend to anyone",
    "Waste of money, the product is poorly made",
    "Very bad quality, extremely disappointed overall",
    "Defective product, returned it immediately for refund",
    "Do not waste your money on this low quality item",
    "Worst product ever, falls apart immediately after use",
    "Awful, stopped working after three days of light use",
    "Completely broken out of the box, very frustrating",
    "Total disappointment, product does not match description",
    "Very poor build quality, not durable at all",
    "Regret buying this, it is cheaply made and useless",
    "Bad product, customer service was unhelpful too",
    "Broken after one use, really bad experience overall",
    "Terrible quality control, arrived cracked and scratched",
    "Would give zero stars if possible, absolutely awful",
    "Disgusting quality, completely not worth the price",
    "Hated this product, returned it within two days",
    "Very disappointing, does not work as promised at all",
    "Cheap and nasty, will never buy from this brand again",
    "Horrible experience, product is faulty and useless",
    "Terrible, packaging was bad and product was broken",
    "Annoying product that malfunctions constantly every day",
    "Very angry about this purchase, total scam product",
    "Poor quality item, nothing like what was advertised",
    "Dreadful product, cheap materials and bad finish",
    "Not worth it at all, breaks easily and feels cheap",
    "Really bad product, falling apart after a few days",
    "Extremely disappointed, this was a total waste",
    "Useless product, do not recommend to anyone at all",
]

NEUTRAL_REVIEWS = [
    "Product is okay, nothing special but does the job",
    "It is average, works fine but not particularly impressive",
    "Decent product for the price, has some minor issues",
    "Neither great nor terrible, just an average product",
    "It works as described, nothing more and nothing less",
    "Acceptable quality, meets basic requirements adequately",
    "Okay product, delivery was on time and packaging fine",
    "Moderate quality, does what it is supposed to do",
    "Not bad but not great either, fairly mediocre overall",
    "Works fine for basic use, nothing exceptional about it",
    "Average product, expected more but it is acceptable",
    "It does what it says, no major issues so far at all",
    "Fair quality for the price, has a few minor drawbacks",
    "Mediocre product, could be better in a few areas",
    "Okay for everyday use, not outstanding in any way",
    "Standard product, gets the job done without issues",
    "It is fine, nothing to complain about or praise",
    "Received the product on time, it works as expected",
    "Not amazing but not bad, a fairly typical product",
    "Reasonable product, meets my basic everyday needs",
    "It is what it is, works adequately for normal use",
    "Average build quality, nothing too impressive here",
    "Fine product, no major complaints or compliments",
    "Works okay for the price, meets expectations barely",
    "Neutral experience, product is functional but basic",
    "Alright product, some good points and some bad ones",
    "Moderate satisfaction with this, nothing special really",
    "The product is acceptable for casual everyday use",
    "So and so, has its pros and cons, nothing standout",
    "Used it a few times, seems fine but time will tell",
    "Just about okay, I have seen better and worse products",
    "Does the job adequately, no outstanding features at all",
    "Good enough for my needs, nothing more to add",
    "Middle of the road product, average in every sense",
    "It functions correctly but design could be improved",
    "Somewhat satisfied, product works but feels ordinary",
    "Nothing wrong but nothing impressive about this either",
    "Passable product, would not go out of my way to buy",
    "Okay quality, arrived on time but feels a bit flimsy",
    "Neutral about this, it works but is not exciting at all",
]

def build_dataset():
    reviews  = POSITIVE_REVIEWS + NEGATIVE_REVIEWS + NEUTRAL_REVIEWS
    labels   = (['Positive'] * len(POSITIVE_REVIEWS) +
                ['Negative'] * len(NEGATIVE_REVIEWS) +
                ['Neutral']  * len(NEUTRAL_REVIEWS))
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Basic stopword removal
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for',
                 'of','with','is','it','this','that','was','are','be','as',
                 'i','my','me','we','our','your','you','he','she','they',
                 'have','has','had','do','did','will','just','so','very',
                 'its','from','by','not','no','after','been','about'}
    tokens = [w for w in text.split() if w not in stopwords]
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────

def perform_eda(df):
    print("=" * 60)
    print("  SENTIMENT ANALYZER — EDA")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['sentiment'].value_counts()}")
    df['review_length'] = df['review'].apply(lambda x: len(x.split()))
    print(f"\nAvg review length: {df['review_length'].mean():.1f} words")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Sentiment Analyzer — EDA', fontsize=16, fontweight='bold')

    # Class distribution
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    counts = df['sentiment'].value_counts()
    bars = axes[0].bar(counts.index, counts.values,
                       color=[colors[c] for c in counts.index], edgecolor='black')
    axes[0].set_title('Sentiment Distribution')
    axes[0].set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(val), ha='center', fontweight='bold')

    # Review length distribution
    for sentiment, color in colors.items():
        subset = df[df['sentiment'] == sentiment]['review_length']
        axes[1].hist(subset, bins=10, alpha=0.6, label=sentiment, color=color)
    axes[1].set_title('Review Length by Sentiment')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Pie chart
    axes[2].pie(counts.values, labels=counts.index,
                colors=[colors[c] for c in counts.index],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[2].set_title('Sentiment Share')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] EDA plots saved: eda_plots.png")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────

def train_and_evaluate(df):
    df['clean_review'] = df['review'].apply(preprocess_text)
    X = df['clean_review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Naive Bayes':          Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
                                          ('clf',   MultinomialNB())]),
        'Logistic Regression':  Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
                                          ('clf',   LogisticRegression(max_iter=1000, random_state=42))]),
        'Linear SVM':           Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
                                          ('clf',   LinearSVC(random_state=42, max_iter=2000))]),
    }

    results = {}
    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        cv      = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {'model': pipeline, 'accuracy': acc,
                         'cv_mean': cv.mean(), 'cv_std': cv.std(),
                         'y_pred': y_pred, 'y_test': y_test}
        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"  Test Accuracy : {acc*100:.2f}%")
        print(f"  CV Score      : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
        print(f"\n{classification_report(y_test, y_pred)}")

    return results, X_train, y_train

# ─────────────────────────────────────────────
# 5. VISUALIZE RESULTS
# ─────────────────────────────────────────────

def plot_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Sentiment Analyzer — Model Results', fontsize=16, fontweight='bold')

    names      = list(results.keys())
    accs       = [results[n]['accuracy']*100 for n in names]
    cv_means   = [results[n]['cv_mean']*100  for n in names]
    cv_stds    = [results[n]['cv_std']*100   for n in names]
    bar_colors = ['#9b59b6', '#3498db', '#e67e22']

    # Accuracy bar chart
    bars = axes[0,0].bar(names, accs, color=bar_colors, edgecolor='black')
    axes[0,0].set_ylim(0, 115)
    axes[0,0].set_title('Test Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy (%)')
    for bar, acc in zip(bars, accs):
        axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                       f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=10)
    axes[0,0].tick_params(axis='x', rotation=12)

    # CV scores
    axes[0,1].bar(names, cv_means, yerr=cv_stds, color=bar_colors, capsize=5, edgecolor='black')
    axes[0,1].set_title('5-Fold Cross-Validation Score')
    axes[0,1].set_ylabel('CV Accuracy (%)')
    axes[0,1].tick_params(axis='x', rotation=12)

    # TF-IDF top features for Logistic Regression
    lr_pipeline = results['Logistic Regression']['model']
    tfidf       = lr_pipeline.named_steps['tfidf']
    clf         = lr_pipeline.named_steps['clf']
    classes     = clf.classes_
    feature_names = np.array(tfidf.get_feature_names_out())
    top_n = 8
    sentiment_colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    for i, cls in enumerate(classes):
        coef     = clf.coef_[i]
        top_idx  = np.argsort(coef)[-top_n:]
        top_feats = feature_names[top_idx]
        top_vals  = coef[top_idx]
        axes[0,2].barh(top_feats, top_vals, alpha=0.6,
                       color=sentiment_colors.get(cls, 'gray'), label=cls)
    axes[0,2].set_title('Top TF-IDF Features\n(Logistic Regression)')
    axes[0,2].set_xlabel('Coefficient Weight')
    axes[0,2].legend(fontsize=8)

    # Confusion matrices
    labels = ['Negative', 'Neutral', 'Positive']
    for idx, name in enumerate(names):
        ax = axes[1, idx]
        cm = confusion_matrix(results[name]['y_test'], results[name]['y_pred'], labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'Confusion Matrix\n{name}', fontsize=10)
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] Model results saved: model_results.png")

# ─────────────────────────────────────────────
# 6. WORD FREQUENCY PLOT
# ─────────────────────────────────────────────

def plot_word_freq(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Top Words by Sentiment Class', fontsize=14, fontweight='bold')
    sentiments = ['Positive', 'Negative', 'Neutral']
    colors     = ['#2ecc71', '#e74c3c', '#3498db']

    for ax, sentiment, color in zip(axes, sentiments, colors):
        subset = df[df['sentiment'] == sentiment]['review'].apply(preprocess_text)
        cv     = CountVectorizer(max_features=10)
        cv.fit_transform(subset)
        freq   = cv.transform(subset).toarray().sum(axis=0)
        words  = cv.get_feature_names_out()
        sorted_idx = np.argsort(freq)[::-1]
        ax.barh([words[i] for i in sorted_idx], [freq[i] for i in sorted_idx],
                color=color, edgecolor='black', alpha=0.85)
        ax.set_title(f'{sentiment} Reviews')
        ax.set_xlabel('Frequency')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('word_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] Word frequency plot saved: word_frequency.png")

# ─────────────────────────────────────────────
# 7. PREDICT NEW REVIEWS
# ─────────────────────────────────────────────

def predict_reviews(model, reviews):
    print("\n" + "=" * 55)
    print("  LIVE PREDICTIONS")
    print("=" * 55)
    cleaned = [preprocess_text(r) for r in reviews]
    preds   = model.predict(cleaned)
    emoji   = {'Positive': '✅', 'Negative': '❌', 'Neutral': '➖'}
    for review, pred in zip(reviews, preds):
        short = review[:55] + '...' if len(review) > 55 else review
        print(f"  {emoji[pred]} [{pred:8s}] \"{short}\"")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n💬  Sentiment Analyzer — NLP & ML Project")
    print("    CSA2001 | VIT Bhopal\n")

    # Build dataset
    df = build_dataset()
    df.to_csv('sentiment_dataset.csv', index=False)
    print(f"[✓] Dataset built: {df.shape[0]} reviews, 3 sentiment classes")

    # EDA
    perform_eda(df)

    # Word frequency
    plot_word_freq(df)

    # Train & evaluate
    results, X_train, y_train = train_and_evaluate(df)

    # Plot results
    plot_results(results)

    # Live predictions
    best = max(results, key=lambda n: results[n]['accuracy'])
    test_reviews = [
        "This is the best product I have ever bought, absolutely love it!",
        "Terrible quality, broke on the first day, very disappointed.",
        "It works okay, nothing special but does what it should.",
        "Amazing experience, fast delivery and superb quality overall.",
        "Not worth the money, very poor build and bad customer service.",
        "Average product, gets the job done but nothing to write home about.",
    ]
    predict_reviews(results[best]['model'], test_reviews)

    print(f"\n✅ Best model: {best} ({results[best]['accuracy']*100:.2f}% accuracy)")
    print("\n   Files generated:")
    print("   sentiment_dataset.csv | eda_plots.png")
    print("   word_frequency.png    | model_results.png")
