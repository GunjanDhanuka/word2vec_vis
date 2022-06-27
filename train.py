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
import stqdm
warnings.filterwarnings("ignore")
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

Word = WordNetLemmatizer()
stop_words = stopwords.words('english')
TEXT_COL = ''

# @st.cache(suppress_st_warning=True)
def train_on_dataframe(df, text_col, PARAMS):
    print('Entered training subroutine..')
    df = df.dropna().reset_index(drop=True)
    TEXT_COL = text_col
    df[TEXT_COL] = df[TEXT_COL].astype("string")
    t = time()
    preprocess(df, TEXT_COL)
    message = 'Time taken to clean up and preprocess data: {} mins'.format(round((time() - t) / 60, 2))
    print(message)
    st.success(message)
    # min_count (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
    sent = [row.split() for row in df[TEXT_COL]]
    phrases = Phrases(sent, min_count=PARAMS['min_count'], progress_per=5000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    highest_freq = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

    st.write("Top 10 words from the text corpus:")
    st.write(highest_freq)

    st.info('Model training has begun!')
    cores = multiprocessing.cpu_count() #Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=PARAMS['min_count'],
                    window=PARAMS['window'],
                    vector_size=PARAMS['vector_size'],
                    sample=6e-5,
                    sg= 1 if PARAMS['sg_choice'] == 'Skip-gram' else 0,
                    alpha=PARAMS['alpha'], 
                    min_alpha=PARAMS['min_alpha'], 
                    negative=PARAMS['negative'],
                    workers=cores-1)
    # w2v_model = Word2Vec(min_count=20,
    #                 window=2,
    #                 vector_size=300,
    #                 sample=6e-5,
    #                 sg=1,
    #                 alpha=0.03, 
    #                 min_alpha=0.0007, 
    #                 negative=20,
    #                 workers=cores-1)
    
    # Building vocabulary table
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    message = 'Time taken to build vocab: {} mins'.format(round((time() - t) / 60, 2))
    print(message)
    st.success(message)
    t = time()
    with st.spinner('Training Word2Vec Model'):
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=PARAMS['epochs'], report_delay=1)
    message = 'Time taken to train the model: {} mins'.format(round((time() - t) / 60, 2))
    st.success(message)
    print(message)
    if 'model' not in st.session_state:
        st.session_state['model'] = w2v_model
    # w2v_model.save('simpsons_w2v.bin')
    return w2v_model

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
    print('Preprocessing initiated...')
    df[TEXT_COL] = df[TEXT_COL].apply(clean)
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: remove_punct(x))
    lower_case(df, TEXT_COL)
    stop_words_remove(df, TEXT_COL)
    lemmatize(df, TEXT_COL)
    print('Preprocessing finished')
    return


def train_on_raw_text(text, PARAMS):
    print('Entered Training Subroutine')
    text = re.sub(r"[^.A-Za-z]", " ", text)
    text = text.lower()
    sentence_list = text.split(".")
    sentences_without_stopword = []
    for sentence in sentence_list:
        temp = []
        for word in sentence.split():
            if word not in stop_words:
                temp.append(word)
        sentences_without_stopword.append(" ".join(temp))

    tokens=[nltk.word_tokenize(words) for words in sentences_without_stopword]
    phrases = Phrases(tokens, min_count=PARAMS['min_count'], progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[tokens]
    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    highest_freq = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

    st.write("Top 10 words from the text corpus:")
    st.write(highest_freq)

    st.info('Model training has begun!')
    cores = multiprocessing.cpu_count() #Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=PARAMS['min_count'],
                    window=PARAMS['window'],
                    vector_size=PARAMS['vector_size'],
                    sample=6e-5,
                    sg= 1 if PARAMS['sg_choice'] == 'Skip-gram' else 0,
                    alpha=PARAMS['alpha'], 
                    min_alpha=PARAMS['min_alpha'], 
                    negative=PARAMS['negative'],
                    workers=cores-1)
    # Building vocabulary table
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    message = 'Time taken to build vocab: {} mins'.format(round((time() - t) / 60, 2))
    print(message)
    st.success(message)
    t = time()
    with st.spinner('Training Word2Vec Model'):
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=PARAMS['epochs'], report_delay=1)
    message = 'Time taken to train the model: {} mins'.format(round((time() - t) / 60, 2))
    st.success(message)
    print(message)
    if 'model' not in st.session_state:
        st.session_state['model'] = w2v_model
    return w2v_model

