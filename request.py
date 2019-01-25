#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:27:27 2019

@author: alkesha
"""

import requests
url = 'http://127.0.0.1:5000/api'
r = requests.post(url,json={'val':'Free entry in 2 a wkly comp to win FA Cup finally '})
print(r.json())