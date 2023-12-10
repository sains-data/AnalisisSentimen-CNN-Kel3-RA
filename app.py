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

    # Use st.text_input instead of st.text_area
    user_input = st.text_input("Masukkan review Anda di sini")

    # Store the previous user input
    prev_user_input = st.session_state.get('prev_user_input', None)

    # Add a condition to rerun the script when user_input changes
    if user_input != prev_user_input:
        st.session_state.prev_user_input = user_input

    if st.button('Prediksi') or user_input != prev_user_input:
        try:
            prediksi = review_prediction(user_input, model, tokenizer)
            if prediksi is not None:
                if prediksi[0][0] > 0.5:
                    st.write('Review Anda adalah review positif')
                else:
                    st.write('Review Anda adalah review negatif')
                
                # Add "Coba Masukkan Review Lagi" button to refresh everything
                if st.button('Coba Masukkan Review Lagi'):
                    st.session_state.prev_user_input = None
            else:
                st.write("Tidak dapat memproses review Anda.")
        except Exception as e:
            st.write("Terjadi kesalahan dalam pemrosesan: ", e)

if __name__ == '__main__':
    main()