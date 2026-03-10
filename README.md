# 🎫 Support Ticket Classification & Prioritization
### Future Interns — Machine Learning Task 2 (2026)

## 📌 Overview
An ML-powered system that automatically classifies customer support tickets
into categories and assigns priority levels — eliminating manual sorting
and helping support teams respond faster.

## 🎯 Objective
- Read raw support ticket text
- Classify into categories (Hardware, Access, Billing, etc.)
- Assign priority level (High / Medium / Low)

## 🛠️ Tech Stack
- **Language:** Python
- **NLP:** NLTK (tokenization, stopword removal, lemmatization)
- **Features:** TF-IDF Vectorization (unigrams + bigrams)
- **Models:** Logistic Regression · Random Forest · Naive Bayes
- **Libraries:** Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## 📁 Project Structure
```
FUTURE_ML_02/
├── data/
│   └── all_tickets_processed_improved_v3.csv
├── outputs/
│   ├── category_confusion_matrix.png
│   ├── priority_confusion_matrix.png
│   └── model_comparison.png
├── support_ticket_classifier.py
└── README.md
```

## ⚙️ How to Run
```bash
# Install dependencies
pip install nltk scikit-learn pandas numpy matplotlib seaborn

# Run the script
python support_ticket_classifier.py
```

## 🔄 ML Pipeline
1. Load dataset (`Document`, `Topic_group` columns)
2. Text cleaning → lowercase, remove URLs/emails, punctuation
3. Tokenization → stopword removal → lemmatization
4. TF-IDF feature extraction (5000 features)
5. Train & compare 3 models for category + priority
6. Evaluate with accuracy, precision, recall, F1-score
7. Save confusion matrices and comparison chart

## 📊 Results
| Task | Best Model | Accuracy |
|------|-----------|----------|
| Category Classification | — | —% |
| Priority Prediction | — | —% |

## 💡 Business Impact
- ✔ Auto-routes tickets to the right team instantly
- ✔ Flags high-priority issues for immediate response
- ✔ Scales to thousands of tickets/day with zero manual effort
- ✔ Consistent decisions, 24/7

## 👤 Author
**Tanam Adithya**
[GitHub](https://github.com/Adithya-76) · [LinkedIn](https://linkedin.com/in/adithya-tanam-al76)