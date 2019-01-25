#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:23:53 2019

@author: alkesha
"""


from flask import Flask, request, jsonify
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer
import matplotlib.pyplot as mlt
import math
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  train_test_split

#preprocessing of messages
def proc(sms, lower_case = True, stem = True, stop_words = True, gram = 2):
    sms= sms.lower()
    words = word_tokenize(sms)
    stopword = stopwords.words('english')
    words = [word for word in words if word not in stopword]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    return words

app = Flask(__name__)
# Load the model
model = pickle.load(open('spam_ham.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    if request.method=='POST':
        return "aaaaaaaaaaaaaaA"
    #return jsonify(output)
if __name__ == '__main__':
    app.run()