# Email / SMS Spam Detection
 
A machine learning classifier that distinguishes spam from legitimate (ham) messages. The project runs a full NLP pipeline — cleaning, EDA, text preprocessing, TF-IDF vectorization — and benchmarks eleven classifiers before combining the strongest into ensemble models. The final TF-IDF vectorizer and model are serialized for deployment.
 
## Overview
 
Spam filtering is a classic binary text-classification problem where **precision matters more than raw accuracy** — wrongly flagging a real message as spam is costly. This project prioritizes precision when comparing models and selects Multinomial Naive Bayes for its strong precision on TF-IDF features.
 
## Dataset
 
A labelled collection of messages tagged as `ham` or `spam`, loaded directly from a GitHub raw URL (`email_spam.csv`).
 
| Column | Description |
| :--- | :--- |
| `target` (originally `v1`) | Label — `ham` or `spam`, encoded to 0/1 |
| `text` (originally `v2`) | Raw message content |
 
Three empty trailing columns are dropped, columns are renamed, the label is encoded with `LabelEncoder`, and duplicate rows are removed. The classes are imbalanced (ham far outnumbers spam).
 
## Pipeline
 
1. **Data cleaning** — drop empty columns, rename `v1`/`v2`, label-encode the target, drop duplicates, check for missing values.
2. **EDA** — class distribution pie chart; engineered features for character, word, and sentence counts; distribution comparisons (histograms, pairplot) between spam and ham.
3. **Text preprocessing** — lowercase, tokenize, keep alphanumeric tokens, remove stopwords and punctuation, apply Porter stemming.
4. **Visualization** — WordClouds and top-30 word frequency bar charts for spam vs. ham corpora.
5. **Model building** — TF-IDF vectorization, 80/20 train/test split, benchmark eleven classifiers on accuracy and precision.
6. **Model improvement** — Voting and Stacking ensembles built from the top performers.
7. **Serialization** — save the TF-IDF vectorizer (`vectorizer.pkl`) and the Naive Bayes model (`model.pkl`) with pickle.
## Models Compared
 
Naive Bayes variants (Gaussian, Multinomial, Bernoulli) plus SVC, K-Nearest Neighbors, Decision Tree, Logistic Regression, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost — each scored on accuracy and precision.
 
Two ensembles are then built:
- **Voting Classifier** (soft) — SVC + Multinomial NB + Extra Trees
- **Stacking Classifier** — same base estimators with a Random Forest meta-learner
**Multinomial Naive Bayes** is selected as the final model for its high precision, and is the one pickled for deployment.
 
## Tech Stack
 
Python · scikit-learn · XGBoost · NLTK · pandas · NumPy · Matplotlib · Seaborn · WordCloud · pickle
 
## Getting Started
 
```bash
pip install scikit-learn xgboost nltk wordcloud pandas numpy matplotlib seaborn
```
 
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
 
Open `Email_SMS_Spam_Detection.ipynb` and run the cells top to bottom. The dataset loads automatically from its GitHub URL, so no manual download is required. Running the final cells produces `vectorizer.pkl` and `model.pkl`.
 
## Using the Saved Model
 
```python
import pickle
 
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
 
# Apply the same preprocess_data() cleaning used in the notebook, then:
vec = tfidf.transform([cleaned_message]).toarray()
prediction = model.predict(vec)   # 0 = ham, 1 = spam
```
 
## Project Structure
 
```
.
├── Email_SMS_Spam_Detection.ipynb   # Main notebook: cleaning, EDA, modeling, ensembles
├── vectorizer.pkl                   # Saved TF-IDF vectorizer (generated)
├── model.pkl                        # Saved Naive Bayes model (generated)
└── README.md
```
