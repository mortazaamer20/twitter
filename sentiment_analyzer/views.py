# sentiment/views.py

import pandas as pd
import nltk
import re
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from django.shortcuts import render



#حملنة المكاتب
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))


def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  
    tweet = re.sub(r'@\w+', '', tweet)  
    tweet = re.sub(r'\$\w+', '', tweet)  
    tweet = re.sub(r'^rt', '', tweet)  
    tweet = re.sub(r'#', '', tweet)  
    tweet = tweet.lower()  
    tweet = tweet.strip()   
    tokens = TweetTokenizer().tokenize(tweet) 
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation ]
    return stem_words(tokens)

def sentiment_analysis(text):
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    positive_df = pd.DataFrame(positive_tweets, columns=['tweet'])
    positive_df['label'] = 1
    negative_df = pd.DataFrame(negative_tweets, columns=['tweet'])
    negative_df['label'] = 0

    all_tweets_df = pd.concat([positive_df, negative_df], ignore_index=True)

    def create_frequency_dict(tweets, labels):
        freq = {}
        for tweet, y in zip(tweets, labels):   
            processed_words = preprocess(tweet)   
            for word in processed_words:        
                pair = (word, y)    
                if pair in freq:
                    freq[pair] += 1
                else:
                    freq[pair] = 1
        return freq 

    freq_dict = create_frequency_dict(all_tweets_df['tweet'].tolist(), all_tweets_df['label'].tolist())

    def count_total_words(tweet):
        positive_count = 0
        negative_count = 0
        for word in tweet:
            positive_count += freq_dict.get((word, 1), 0)
            negative_count += freq_dict.get((word, 0), 0)
        return [positive_count, negative_count]

    pro_tweet = preprocess(text)  
    positive_count, negative_count = count_total_words(pro_tweet)

    
    sentiment_df = pd.DataFrame([[positive_count, negative_count]], columns=['Positive Count', 'Negative Count'])
    sentiment_df['class'] = [1 if positive_count > negative_count else 0]

    X = sentiment_df[['Positive Count', 'Negative Count']]
    y = sentiment_df['class']

    gnb = GaussianNB()
    gnb.fit(X, y)
    
    y_pred = gnb.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return sentiment_df['class'][0], accuracy, precision, recall, f1



def home(request):
    # هنا نخلي التويت القديمة تظهر في الصفحة
    if 'tweet_history' not in request.session:
        request.session['tweet_history'] = []
    
    # ناخذ النص من اليوزر ونعمل له تحليل
    if request.method == 'POST':
        text = request.POST.get('text', '')  
        if text:
            #نحلل النص بواسطة الدالة sentiment_analysis
            sentiment, accuracy, precision, recall, f1 = sentiment_analysis(text)
            
            # التأكد من ان القيم هي ارقام بلغة بايثون 
            sentiment = int(sentiment) #تحويل الى رقم صحيح int
            accuracy = float(accuracy)  # تحويل الى رقم عشري float
            precision = float(precision)  # تحويل الى رقم عشري float
            recall = float(recall)  # تحويل الى رقم عشري float
            f1 = float(f1)  # تحويل الى رقم عشري float
            
            tweet_data = {
                'text': text,
                'sentiment': sentiment,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # اضافة التويت الجديد الى التاريخ
            tweet_history = request.session['tweet_history']
            tweet_history.append(tweet_data)
            request.session['tweet_history'] = tweet_history 
    
    # اعادة ترتيب التويتات بحيث يكون اخر تويت هو الاول
    tweet_history = request.session.get('tweet_history', [])
    tweet_history.reverse()

    # اظهار الصفحة
    return render(request, 'home.html', {'tweet_history': tweet_history})