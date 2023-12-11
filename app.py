import streamlit as st
import numpy as np
import tensorflow as tf
import contractions
import re
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pickle

# Load model
model = tf.keras.models.load_model('model_DL.h5')

# Tokenizer initialization
max_features = 1000
max_len = 100
tokenizer = Tokenizer(num_words=max_features, split=' ')

# Loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define Streamlit app
st.title("Binary Text Classification Amazon Reviews App (sentimen)")
st.caption('Hanya tersedia dalam review bahasa Inggris')

st.image('download.png', caption='KELOMPOK 3 RA')

def remove_accented_chars(text):
    """
    Removes accented characters from text.
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def review_prediction(review, model, tokenizer):
    """
    Predicts the sentiment of the review.
    """
    # Preprocessing
    doc = review.translate(review.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()
    norm_docs = [doc]

    # Tokenizing
    X = tokenizer.texts_to_sequences(norm_docs)
    X = sequence.pad_sequences(X, maxlen=max_len)

    # Predicting
    pred = model.predict(X)
    pred = pred[0][0]
    return pred

# Streamlit UI
user_input = st.text_input("Enter your review:")
if st.button("Predict"):
    try:
        prediction = review_prediction(user_input, model, tokenizer)
        prediction = float(prediction)
        if prediction >= 0.5:
            st.write('Your review is positive.')
            st.write('Prediction score:', prediction)
        else:
            st.write('Your review is negative.')
            st.write('Prediction score:', prediction)
    except Exception as e:
        st.error("An error occurred during processing: {}".format(e))
