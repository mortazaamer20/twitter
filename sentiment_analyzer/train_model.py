import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset (replace with the actual file path)
df = pd.read_csv('C:/Users/murtadha/Desktop/sentiment_project/sentiment_analyzer/sentiment_dataset.csv')

# Drop rows with missing sentiment labels
df = df.dropna(subset=['label'])  # Ensure 'label' column exists and is correct

# Sample a smaller subset for testing (optional, for memory issues)
df_sample = df.sample(n=1000, random_state=42)  # Adjust the sample size as needed

# Features (X) and target (y)
X = df_sample['tweet']
y = df_sample['label']

# Convert text data into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)  # Limit number of features
X_vec = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Initialize the MultinomialNB model (better for text classification)
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict using the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved!")
