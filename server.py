#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:23:53 2019

@author: alkesha
"""


from flask import Flask, request, jsonify
from flask import render_template, request
import pickle


model = pickle.load(open('model/spam_ham.pickle','rb'))
app = Flask(__name__)
# Load the model
@app.route('/spam_ham', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = str(request.form.get("message"))
        message= [message]
        result=model.predict(message)
        result=result.tolist()
        result=int(result[0])
        print(result)
        if result==0 :
            value="your message is NOT spam  :)"
            print(value)
        else:
            value="your message is SPAM !!! :("
            print(value)
        return render_template('index.html',value=value,result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


