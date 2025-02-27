import joblib
import pickle
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load pretrained model
model = joblib.load('sentiment_model.pkl')

# Load frequency dictionary
with open('frequency_dict.pkl', 'rb') as f:
    freq_dict = pickle.load(f)

stop_words = set(nltk.corpus.stopwords.words('english'))

def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'\$\w+', '', tweet)
    tweet = re.sub(r'^rt', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = tweet.lower().strip()
    
    tokens = TweetTokenizer().tokenize(tweet)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return stem_words(tokens)

def predict_sentiment(text):
    processed_tokens = preprocess(text)
    pos_count = sum(freq_dict.get((word, 1), 0) for word in processed_tokens)
    neg_count = sum(freq_dict.get((word, 0), 0) for word in processed_tokens)
    prediction = model.predict([[pos_count, neg_count]])[0]
    return "Positive" if prediction == 1 else "Negative", pos_count, neg_count
