import streamlit as st
import numpy as np
import tensorflow as tf
import contractions
import re
import tqdm
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


# Load model outside of the function for efficiency
model = tf.keras.models.load_model('model_DL.h5')

# Tokenizer initialization
max_features = 1000
max_len = 100
tokenizer = Tokenizer(num_words=max_features, split=' ')

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
    return pred

def main():
    st.title('App Prediksi Sentimen Review Amazon (hanya tersedia dalam bahasa inggris)')

    user_input = st.text_area("Masukkan review Anda di sini")

    if st.button('Prediksi'):
        try:
            prediksi = review_prediction(user_input, model, tokenizer)
            st.write(prediksi)
            if prediksi[0][0] > prediksi[0][1]:
                st.write('Review Anda adalah review negatif')
            else:
                st.write('Review Anda adalah review positif')

if __name__ == '__main__':
    main()
