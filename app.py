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


model = tf.keras.models.load_model('model_DL.h5')

# def pre_process_corpus(docs):
#   norm_docs = []
#   for doc in tqdm.tqdm(docs):
#     doc = doc.translate(doc.maketrans("\n\t\r", "   "))
#     doc = doc.lower()
#     doc = remove_accented_chars(doc)
#     doc = contractions.fix(doc)
#     # lower case and remove special characters\whitespaces
#     doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
#     doc = re.sub(' +', ' ', doc)
#     doc = doc.strip()
#     norm_docs.append(doc)

#   return norm_docs


#creating function predict with user input preprocessing
def review_prediction(review):
    # Preprocessing
    norm_docs = []
    doc = review.translate(review.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()
    norm_docs.append(doc)

    # Tokenizing
    max_features = 2000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(norm_docs)
    X = tokenizer.texts_to_sequences(norm_docs)
    X = sequence.pad_sequences(X, maxlen=max_len)

    # Predicting
    pred = model.predict(X)
    return pred




def main():
    st.title('App Prediksi Sentimen Review (hanya tersedia dalam bahasa inggris)')

    # with open("download.png", "rb") as file:
    #   st.image(file, caption='Sentiment Review Amazon app')

    # Input review
    user_input = st.text_area("Masukkan review Anda di sini")

    prediksi=''
    if st.button('Prediksi'):
        prediksi = review_prediction([user_input])
        if prediksi[0][0] > 0.5:
            st.write('Review Anda adalah review positif')
        else:
            st.write('Review Anda adalah review negatif')

if __name__ == '__main__':
    main()
