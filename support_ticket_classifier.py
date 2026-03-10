import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── NLTK downloads ────────────────────────────────────────────
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

# =============================================================
# 1. LOAD DATASET
# =============================================================

KAGGLE_FILE = 'data/all_tickets_processed_improved_v3.csv'

if os.path.exists(KAGGLE_FILE):
    df_raw = pd.read_csv(KAGGLE_FILE)
    print(f"✅ Kaggle dataset loaded: {df_raw.shape[0]} rows")
    # Map Kaggle columns to standard names
    df = df_raw[['Document', 'Topic_group']].copy()
    df.columns = ['ticket_text', 'category']

# Auto-assign priority based on category
    priority_map = {
    'Hardware':        'High',
    'Technical Issue': 'High',
    'Network':         'High',
    'Access':          'Medium',
    'Account':         'Medium',
    'Billing':         'Medium',
    'General Query':   'Low',
    'HR':              'Low',
}
    df['priority'] = df['category'].map(priority_map).fillna('Medium')
else:
    print("⚠️  Kaggle CSV not found. Generating synthetic dataset...")
    random.seed(42)

    templates = {
        'Billing': [
            "I was charged twice for my subscription and need a refund.",
            "My invoice shows an incorrect amount. Please fix it.",
            "I cancelled my plan but was still billed.",
            "Payment failed but money was deducted from my account.",
            "Why was I charged a late fee? I paid on time.",
            "Promo code was not applied to my bill.",
            "I need my billing history for the last 6 months.",
        ],
        'Technical Issue': [
            "The app crashes every time I try to upload a file.",
            "Login page is not loading. I keep getting a 500 error.",
            "API integration is broken after your last update.",
            "Two-factor authentication is not sending OTP.",
            "The mobile app crashes on startup on Android 14.",
            "Search functionality is not returning relevant results.",
            "The integration with Slack stopped working yesterday.",
        ],
        'Account': [
            "I forgot my password and the reset email is not arriving.",
            "I need to change the email address on my account.",
            "My account was locked after too many login attempts.",
            "How do I add a team member to my account?",
            "I want to delete my account and all associated data.",
            "I cannot access my account since I changed my phone number.",
        ],
        'General Query': [
            "What are your business hours for customer support?",
            "Do you offer a free trial for the premium plan?",
            "What is the difference between Basic and Pro plans?",
            "Is your service available in India?",
            "What payment methods do you accept?",
            "Is there a student discount available?",
        ],
    }

    priority_weights = {
        'Billing':         ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Technical Issue': ['High', 'High', 'High', 'Medium', 'Low'],
        'Account':         ['Medium', 'Medium', 'High', 'Low', 'Low'],
        'General Query':   ['Low', 'Low', 'Medium', 'Low', 'Low'],
    }

    suffixes = ["Please help ASAP.", "Urgent.", "Awaiting your response.", "This is frustrating.", "Thank you."]
    rows = []
    for category, texts in templates.items():
        for _ in range(150):
            text = random.choice(texts)
            if random.random() > 0.5:
                text += " " + random.choice(suffixes)
            rows.append({
                'ticket_text': text,
                'category':    category,
                'priority':    random.choice(priority_weights[category])
            })

    df = pd.DataFrame(rows)
    print(f"✅ Synthetic dataset created: {len(df)} tickets")

df.dropna(subset=['ticket_text', 'category', 'priority'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nCategory distribution:\n{df['category'].value_counts()}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")

# =============================================================
# 2. TEXT PREPROCESSING
# =============================================================

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)   # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)             # Remove emails
    text = re.sub(r'[^a-z\s]', '', text)            # Keep letters only
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['ticket_text'].apply(clean_text)
print("\n✅ Text preprocessing complete.")

# =============================================================
# 3. TF-IDF FEATURE EXTRACTION
# =============================================================

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X = tfidf.fit_transform(df['clean_text'])
print(f"\nTF-IDF matrix shape: {X.shape}")

# =============================================================
# 4. HELPER — TRAIN & EVALUATE MODELS
# =============================================================

def train_and_evaluate(X, y, label_encoder, task_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes':         MultinomialNB(),
    }

    print(f"\n{'='*55}")
    print(f"  {task_name}")
    print(f"{'='*55}")

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': acc, 'y_pred': y_pred}
        print(f"  {name:<25} Accuracy: {acc*100:.2f}%")

    best_name  = max(results, key=lambda k: results[k]['accuracy'])
    best       = results[best_name]
    print(f"\n  🏆 Best: {best_name} ({best['accuracy']*100:.2f}%)")

    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(
        y_test, best['y_pred'],
        target_names=label_encoder.classes_
    ))

    return best_name, best['model'], results, X_test, y_test

# =============================================================
# 5. CATEGORY CLASSIFICATION
# =============================================================

le_cat  = LabelEncoder()
y_cat   = le_cat.fit_transform(df['category'])

best_cat_name, best_cat_model, cat_results, Xc_test, yc_test = train_and_evaluate(
    X, y_cat, le_cat, "CATEGORY CLASSIFICATION"
)

# =============================================================
# 6. PRIORITY PREDICTION
# =============================================================

le_pri  = LabelEncoder()
y_pri   = le_pri.fit_transform(df['priority'])

best_pri_name, best_pri_model, pri_results, Xp_test, yp_test = train_and_evaluate(
    X, y_pri, le_pri, "PRIORITY PREDICTION"
)

# =============================================================
# 7. VISUALIZATIONS
# =============================================================

def save_confusion_matrix(y_true, y_pred, labels, title, filename, cmap):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {filename}")

print("\n📊 Saving visualizations...")

# Confusion matrices
save_confusion_matrix(
    yc_test, best_cat_model.predict(Xc_test), le_cat.classes_,
    f"Category Confusion Matrix ({best_cat_name})",
    "category_confusion_matrix.png", "Blues"
)
save_confusion_matrix(
    yp_test, best_pri_model.predict(Xp_test), le_pri.classes_,
    f"Priority Confusion Matrix ({best_pri_name})",
    "priority_confusion_matrix.png", "Oranges"
)

# Model comparison bar chart
model_names = list(cat_results.keys())
cat_accs    = [cat_results[m]['accuracy'] for m in model_names]
pri_accs    = [pri_results[m]['accuracy'] for m in model_names]

x     = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width/2, cat_accs, width, label='Category', color='#4F86C6')
b2 = ax.bar(x + width/2, pri_accs, width, label='Priority',  color='#F4A261')
for bar in [*b1, *b2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → model_comparison.png")

# =============================================================
# 8. LIVE PREDICTION FUNCTION
# =============================================================

def predict_ticket(ticket_text: str) -> dict:
    """Classify a raw support ticket into category + priority."""
    cleaned = clean_text(ticket_text)
    vector  = tfidf.transform([cleaned])

    cat_idx   = best_cat_model.predict(vector)[0]
    category  = le_cat.inverse_transform([cat_idx])[0]
    cat_conf  = dict(zip(le_cat.classes_,
                         [round(p, 3) for p in best_cat_model.predict_proba(vector)[0]]))

    pri_idx   = best_pri_model.predict(vector)[0]
    priority  = le_pri.inverse_transform([pri_idx])[0]
    pri_conf  = dict(zip(le_pri.classes_,
                         [round(p, 3) for p in best_pri_model.predict_proba(vector)[0]]))

    return {
        'ticket':             ticket_text,
        'category':           category,
        'category_confidence': cat_conf,
        'priority':           priority,
        'priority_confidence': pri_conf,
    }

# =============================================================
# 9. DEMO PREDICTIONS
# =============================================================

test_tickets = [
    "I was charged twice this month. Please refund the duplicate payment immediately!",
    "The app crashes every time I try to log in on my iPhone.",
    "I forgot my password and the reset link never arrives in my inbox.",
    "Do you offer a student discount for the Pro plan?",
    "API returning 503 errors — my entire integration is down. Critical!",
]

priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}

print(f"\n{'='*65}")
print("  LIVE PREDICTIONS")
print(f"{'='*65}")

for ticket in test_tickets:
    r = predict_ticket(ticket)
    print(f"\n🎫 {r['ticket']}")
    print(f"   📂 Category : {r['category']}")
    print(f"      Confidence: {r['category_confidence']}")
    print(f"   🚦 Priority : {priority_emoji.get(r['priority'], '')} {r['priority']}")
    print(f"      Confidence: {r['priority_confidence']}")

# =============================================================
# 10. FINAL SUMMARY
# =============================================================

print(f"\n{'='*65}")
print("  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  Dataset size          : {len(df)} tickets")
print(f"  TF-IDF features       : {X.shape[1]}")
print(f"  Ngram range           : (1, 2)")
print(f"  Category model        : {best_cat_name}")
print(f"  Category accuracy     : {cat_results[best_cat_name]['accuracy']*100:.2f}%")
print(f"  Priority model        : {best_pri_name}")
print(f"  Priority accuracy     : {pri_results[best_pri_name]['accuracy']*100:.2f}%")
print(f"{'='*65}")
print("""
💡 Business Impact:
  ✔ Auto-routing tickets to the right team instantly
  ✔ High-priority issues flagged for immediate response
  ✔ Scales to thousands of tickets/day with zero manual sorting
  ✔ Consistent decisions, 24/7 — no human fatigue
""")