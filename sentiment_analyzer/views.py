# sentiment_app/sentiment_model.py

import os
import re
import string
import pickle
import pandas as pd
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))


def custom_preprocessor(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'\$\w+', '', tweet)
    tweet = re.sub(r'^rt', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet.lower().strip()


def custom_tokenizer(tweet):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')


positive_df = pd.DataFrame(positive_tweets, columns=['tweet'])
positive_df['label'] = 1
negative_df = pd.DataFrame(negative_tweets, columns=['tweet'])
negative_df['label'] = 0

all_tweets_df = pd.concat([positive_df, negative_df], ignore_index=True)


X = all_tweets_df['tweet']
y = all_tweets_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


EVALUATION_METRICS = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}


PICKLE_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')


with open(PICKLE_PATH, 'wb') as f:
    pickle.dump((model, vectorizer), f)

def load_model():
    
    with open(PICKLE_PATH, 'rb') as f:
        loaded_model, loaded_vectorizer = pickle.load(f)
    return loaded_model, loaded_vectorizer


_model, _vectorizer = load_model()

def predict_tweet(tweet):
    tweet_vec = _vectorizer.transform([tweet])
    prediction = _model.predict(tweet_vec)[0]
    probability = _model.predict_proba(tweet_vec)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, probability

def sentiment_view(request):
    
    if 'tweet_history' not in request.session:
        request.session['tweet_history'] = []

    if request.method == 'POST':
        tweet_text = request.POST.get('text', '')
        if tweet_text:
            
            sentiment, probability = predict_tweet(tweet_text)

            
            tweet_entry = {
                'text': tweet_text,
                'sentiment': sentiment,
                'accuracy': EVALUATION_METRICS.get('accuracy', 0),
                'precision': EVALUATION_METRICS.get('precision', 0),
                'recall': EVALUATION_METRICS.get('recall', 0),
                'f1': EVALUATION_METRICS.get('f1', 0),
            }

            
            tweet_history = request.session['tweet_history']
            tweet_history.insert(0, tweet_entry)

            
            tweet_history = tweet_history[:10]

            
            request.session['tweet_history'] = tweet_history

   
    context = {
        'tweet_history': request.session.get('tweet_history', [])
    }

    return render(request, 'home.html', context)