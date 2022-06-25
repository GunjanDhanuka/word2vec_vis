import re # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from tqdm.notebook import tqdm
import nltk
import string
import os
import re
import spacy  # For preprocessing
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import warnings
import multiprocessing
import streamlit as st
warnings.filterwarnings("ignore")
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

Word = WordNetLemmatizer()
stop_words = stopwords.words('english')
TEXT_COL = ''

def train_on_dataframe(df, text_col):
    df = df.dropna().reset_index(drop=True)
    TEXT_COL = text_col
    df[TEXT_COL] = df[TEXT_COL].astype("string")
    t = time()
    while st.spinner('Cleaning and preprocessing data...'):
        preprocess(df, TEXT_COL)
    message = 'Time taken to clean up and preprocess data: {} mins'.format(round((time() - t) / 60, 2))
    print(message)
    st.success(message)
    # min_count (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
    sent = [row.split() for row in df[TEXT_COL]]
    phrases = Phrases(sent, min_count=20, progress_per=5000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    highest_freq = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

    st.write("Top 10 words from the text corpus:")
    st.write(highest_freq)

    min_count = st.sidebar.slider('Ignores all words with total frequency lower than this', 1, 50, 20)
    window = st.sidebar.slider('Maximum distance between the current and predicted word within a sentence.', 1, 10, 2)
    sg_choice = st.sidebar.selectbox('Training Algorithm', ('Skip-gram', 'CBOW'))
    vector_size = st.sidebar.slider('Dimensionality of the word vectors', 50, 300, 300)
    alpha = st.sidebar.slider('Initial learning rate', 0.001, 0.1, 0.03)
    min_alpha = st.sidebar.slider('Learning rate with linearly drop to min_alpha as training progresses', 0.00001, 0.001, 0.0007)
    negative = st.sidebar.slider('If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn', 5, 20, 20)
    epochs = st.sidebar.slider('Number of epochs for training (more the epochs, greater the training time)', 5, 30, 10)

    sg = 1 if sg_choice == "Skip-gram" else 0

    cores = multiprocessing.cpu_count() #Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=min_count,
                     window=window,
                     vector_size=vector_size,
                     sample=6e-5,
                     sg=sg,
                     alpha=alpha, 
                     min_alpha=min_alpha, 
                     negative=negative,
                     workers=cores-1)
    
    # Building vocabulary table
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    message = 'Time to build vocab: {} mins'.format(round((time() - t) / 60, 2))
    print(message)
    st.success(message)


    if st.button('Train Word2Vec Model'):
        t = time()
        with st.spinner('Training Word2Vec Model'):
            w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)
        message = 'Time taken to train the model: {} mins'.format(round((time() - t) / 60, 2))
        st.success(message)
        print(message)
        return w2v_model
    else:
        st.sidebar.write('Press the button to start training!')

def clean(raw):
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
    result = re.sub('&gt;', "", result) # greater than sign
    result = re.sub('&#x27;', "'", result) # apostrophe
#     result = re.sub('&quot;', '"', result) 
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result) # paragraph tag
    result = re.sub('<i>', ' ', result) #italics tag
    result = re.sub('</i>', '', result) 
    result = re.sub('&#62;', '', result)
    result = re.sub("\n", '', result) # newline 
    return result

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', text)
    return text

def lower_case(df, TEXT_COL):
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: " ".join(x.lower() for x in x.split()))

def lemmatize(df, TEXT_COL):
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: " ".join([Word.lemmatize(word) for word in x.split()]))

def stop_words_remove(df, TEXT_COL):
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

def preprocess(df, TEXT_COL):
    df[TEXT_COL] = df[TEXT_COL].apply(clean)
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: remove_punct(x))
    lower_case(df, TEXT_COL)
    stop_words_remove(df, TEXT_COL)
    lemmatize(df, TEXT_COL)

