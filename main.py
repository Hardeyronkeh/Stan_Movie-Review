import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# 1. Load and preprocess data
def load_data(directory):
    data = {'review': [], 'sentiment': []}
    for sentiment in ['pos', 'neg']:
        path = os.path.join(directory, sentiment)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                data['review'].append(f.read())
                data['sentiment'].append(1 if sentiment == 'pos' else 0)
    return pd.DataFrame(data)

train_data = load_data('train')
test_data = load_data('test')

# Convert reviews to lowercase
train_data["review"] = train_data["review"].str.lower()
test_data["review"] = test_data["review"].str.lower()

# Remove punctuation
def remove_punctuation(text):
    return ''.join([ch for ch in text if ch not in string.punctuation])

train_data["review"] = train_data["review"].apply(remove_punctuation)
test_data["review"] = test_data["review"].apply(remove_punctuation)

# Tokenize reviews
train_data["review"] = train_data["review"].apply(word_tokenize)
test_data["review"] = test_data["review"].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

train_data["review"] = train_data["review"].apply(remove_stopwords)
test_data["review"] = test_data["review"].apply(remove_stopwords)

# Convert tokens back to strings for vectorization
train_data["review"] = [' '.join(review) for review in train_data["review"]]
test_data["review"] = [' '.join(review) for review in test_data["review"]]

# 2. Vectorization using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
train_tfidf = vectorizer.fit_transform(train_data['review'])
test_tfidf = vectorizer.transform(test_data['review'])

# 3. Train Logistic Regression model
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(train_tfidf, train_data['sentiment'])

# 4. Evaluate Logistic Regression model
lr_predictions = lr_model.predict(test_tfidf)
lr_accuracy = accuracy_score(test_data['sentiment'], lr_predictions)

print(f'Logistic Regression Accuracy: {lr_accuracy}')

# 5. Save the trained Logistic Regression model and vectorizer
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Logistic Regression model and vectorizer saved successfully.")
