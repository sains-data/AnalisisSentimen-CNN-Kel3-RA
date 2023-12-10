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

def pre_process_corpus(docs):
  norm_docs = []
  for doc in tqdm.tqdm(docs):
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()
    norm_docs.append(doc)

  return norm_docs


#creating function predict with user input preprocessing
def review_prediction(input_data):
  norm_input_data = pre_process_corpus(input_data)
  tokenizer = Tokenizer(num_words=5000)
  tokenizer.fit_on_texts(norm_input_data)
  input_data = tokenizer.texts_to_sequences(norm_input_data)
  input_data = sequence.pad_sequences(input_data, maxlen=500)
  prediction = model.predict(input_data)
  return prediction

    


def main():
    st.title('App Prediksi Sentimen Review')

    # with open("download.png", "rb") as file:
    #   st.image(file, caption='Sentiment Review Amazon app')


    # Input review
    user_input = st.text_area("Masukkan review Anda di sini")

    if st.button('Prediksi'):
        # Prediksi review
        pred = review_prediction([user_input])
        pred_label = np.argmax(pred)

        # Tampilkan hasil
        if pred_label == 0:
            st.write('Review Anda: ', user_input)
            st.write('Hasil prediksi: Negative')
        else :
            st.write('Review Anda: ', user_input)
            st.write('Hasil prediksi: Positive')
        
        # Tampilkan probabilitas
        st.write('Probabilitas: ', pred)


if __name__ == '__main__':
    main()
