import streamlit as st
import numpy as np
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import re
import tensorflow as tf

# Download NLTK data at startup
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)  # Required for lemmatization
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Load model and vectorizer with proper error handling
@st.cache_resource  # Cache the model loading
def load_models():
    try:
        model = keras.models.load_model('Model-Nlp-full-v2.keras', compile=False)
        tfid = joblib.load('tfidf_vectorizer_full-v2.joblib')
        return model, tfid
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, tfid = load_models()

# Initialize NLP tools
stem = PorterStemmer()
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def predict_review(text):
    try:
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
            
        # Text cleaning
        cleaned_review = re.sub("<.*?>", "", text)
        cleaned_review = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        cleaned_review = cleaned_review.lower()
        
        # Tokenization
        tokenized_review = word_tokenize(cleaned_review)
        if not tokenized_review:
            raise ValueError("Text could not be tokenized")
            
        # Text processing
        filtered_text = [word for word in tokenized_review if word not in stop_words]
        stemmed_review = [stem.stem(word) for word in filtered_text]
        processed_text = ' '.join(stemmed_review)
        
        # Vectorization
        if not tfid:
            raise ValueError("TF-IDF vectorizer not loaded")
        tfid_review = tfid.transform([processed_text])
        
        # Prediction
        if not model:
            raise ValueError("Model not loaded")
        sentiment_prediction = model.predict(tfid_review)
        
        return np.argmax(sentiment_prediction)
        
    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
        return None

# Streamlit UI
st.title("NLP Chat Bot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

reaction_dict = {
    0: 'A negative sentiment detected 😞',
    1: 'A neutral sentiment detected 😐',
    2: 'A positive sentiment detected 😊',
}

# Handle user input
prompt = st.chat_input("What's your opinion?")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display prediction
    prediction = predict_review(prompt)
    if prediction is not None:
        response = reaction_dict.get(prediction, "Unable to determine sentiment")
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
