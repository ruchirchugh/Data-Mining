# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:59:00 2019

@author: ruchir
"""
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# first covert to lower case, second we remove punctuation, third we remove apostrophe
def preprocess(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    data = np.char.lower(data)
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = np.char.replace(data, "'", "")
    return data

def tokenizing(query):
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(query)
    highlight = []
    for t in tokens:
        if t not in stopword:
            highlight.append(lemmatizer.lemmatize(t).lower())
    return highlight

def image_search(query):
    image_read()
    preprocessed = preprocess(query)
    highlight = tokenizing(query)
    tokens = word_tokenize(str(preprocessed))
    print(tokens)
    img = pd.read_csv("annotation.csv")
    scores = pickle.load(open("imgtfidf.p", "rb"))
    values = {}
    for i in scores:
        if i[1] in tokens:
            try:
                values[i[0]] += scores[i]
            except:
                values[i[0]] = scores[i]
    values = sorted(values.items(), key=lambda x: x[1], reverse=True)
    imageNumber = []
    value = []
    for i in values[:20]:
        imageNumber.append(i[0])
        value.append(i[1])
    result = []
    j = 0
    for i in imageNumber:
        result.append([img['caption'][i], img['url'][i],value[j]])
        j += 1
    pd.set_option('display.max_columns', -1)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)

    print("***DONE***")

    return result , highlight

def image_read():
    img = pd.read_csv("annotation.csv")
    processed = []
    for i in range(len(img)):
        processed.append(word_tokenize(str(preprocess(img['caption'][i]))))
    frequency = {}
    length = len(img)
    for i in range(length):
        for w in processed[i]:
            try:
                frequency[w].add(i)
            except:
                frequency[w] = {i}
    for i in frequency:
        frequency[i] = len(frequency[i])

    def documentFrequency(word):
        count = 0
        try:
            count = frequency[word]
        except:
            pass
        return count

    doc = 0
    totatlScore = {}
    for i in range(length):
        letters = processed[i]
        counter = Counter(letters + processed[i])
        w_count = len(letters + processed[i])
        for token in np.unique(letters):
            tf = counter[token] / w_count
            df = documentFrequency(token)
            idf = np.log((length + 1) / (df + 1))
            totatlScore[doc, token] = tf * idf
        doc += 1

    for i in totatlScore:
        totatlScore[i] *= 0.3
    pickle.dump(totatlScore, open("imgtfidf.p", "wb"))