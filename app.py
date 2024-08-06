import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer
model_path = 'svm_model.pkl'
vectorizer_path = 'vectorizer.pkl'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Define text preprocessing function
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to predict sentiment
def predict_sentiment(text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Streamlit app
st.title('Sentiment Analysis Web App')

# User input
user_input = st.text_area("Enter a review:", "")

# Predict sentiment
if st.button('Predict'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f'The sentiment of the review is: **{sentiment}**')
    else:
        st.write('Please enter a review to analyze.')

# Display some example reviews
st.sidebar.header('Example Reviews')
examples = [
    "Wow... Loved this place.",
    "Crust is not good.",
    "Not tasty and the texture was just nasty.",
    "Stopped by during the late May bank holiday of...",
    "The selection on the menu was great and so were the dishes."
]

for example in examples:
    if st.sidebar.button(f"Analyze: {example}"):
        sentiment = predict_sentiment(example)
        st.sidebar.write(f'The sentiment of the review is: **{sentiment}**')

