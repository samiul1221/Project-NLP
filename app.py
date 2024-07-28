import streamlit as st
import numpy as np
from tensorflow import keras

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import re
import tensorflow as tf

model = keras.models.load_model('Model-Nlp-full-v2.keras',compile=False)

# tfid = TfidfVectorizer()

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

    
stem = PorterStemmer()
lemma = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
tfid = joblib.load('tfidf_vectorizer_full-v2.joblib')



    
def predict_review(text):
    cleaned_review = re.sub("<.*?>","",text)
    cleaned_review = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_text = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stem.stem(word) for word in filtered_text]
    tfid_review = tfid.transform([' '.join(stemmed_review) ])
    sentiment_prediction = model.predict(tfid_review)
    # 0 for -ve, 1 for neutral, 2 for +ve
    return np.argmax(sentiment_prediction)

st.title("NLP Chat Bot")

if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

reactin_dict = {
    0: 'A negative Sentiment detected',
    1: 'A neutral Sentiment detected',
    2: 'A positive Sentiment detected'
}
# React to user input
prompt = st.chat_input("Whats Your Opinion?")
try:
    if prompt:
    
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
    
        # Now for the response
        response = reactin_dict[predict_review(prompt)]
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
    
        st.session_state.messages.append({'role':"assistant","content":response})
except :
    st.error("Something went wrong... Pls try again")    
