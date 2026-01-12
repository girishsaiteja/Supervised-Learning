
# Import required libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

twitter = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/TwitterDataset.csv")
# Display dataset information
print(twitter.info())
print(twitter.head())

# Convert text column to string
twitter['text'] = twitter['text'].astype(str)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()                                   # Convert to lowercase
    text = re.sub(r'\d+', '', text)                       # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'[@/#|]', ' ', text)                   # Replace @, /, | with space
    text = re.sub(r'\s+', ' ', text).strip()              # Remove extra whitespace
    return text

# Apply cleaning
twitter['clean_text'] = twitter['text'].apply(clean_text)

# Create Document-Term Matrix
vectorizer = CountVectorizer(stop_words='english', min_df=5)
X = vectorizer.fit_transform(twitter['clean_text'])
y = twitter['sentiment']

# Split into train/test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print(f"\nAccuracy: {accuracy_score(y_test, pred) * 100:.2f}%")

# Naive Bayes with Laplace Smoothing (like laplace=1)
model_laplace = MultinomialNB(alpha=1)
model_laplace.fit(X_train, y_train)
pred_laplace = model_laplace.predict(X_test)

print("\nConfusion Matrix (Laplace=1):\n", confusion_matrix(y_test, pred_laplace))
print("\nAccuracy with Laplace Smoothing: {:.2f}%".format(accuracy_score(y_test, pred_laplace) * 100))

# Word Clouds for Visualization
def show_wordcloud(text_data, title):
    text = " ".join(text_data)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

# Overall Word Cloud
show_wordcloud(twitter['clean_text'], 'All Tweets')

# Word Clouds for Sentiments
if 'Positive' in twitter['sentiment'].unique():
    show_wordcloud(twitter[twitter['sentiment'] == 'Positive']['clean_text'], 'Positive Tweets')

if 'Negative' in twitter['sentiment'].unique():
    show_wordcloud(twitter[twitter['sentiment'] == 'Negative']['clean_text'], 'Negative Tweets')

if 'Neutral' in twitter['sentiment'].unique():
    show_wordcloud(twitter[twitter['sentiment'] == 'Neutral']['clean_text'], 'Neutral Tweets')
