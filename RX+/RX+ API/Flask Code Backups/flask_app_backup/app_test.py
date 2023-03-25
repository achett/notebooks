# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:27:02 2021

@author: mjichkar
"""

import requests
import json
import pandas as pd
import re

url_aam = 'http://localhost:5000/api/aam'
url_ctv = 'http://localhost:5000/api/ctv'
url_pathseg = 'http://localhost:5000/api/pathseg'

data = [[51,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29.32,0,1,0,0,0]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r_coxph = requests.post(url_coxph, data=j_data, headers=headers)
coxph_result = r_coxph.text
print(coxph_result)

r_aam = requests.post(url_aam, data=j_data, headers=headers)
aam_result = r_aam.text
print(aam_result)

r_ctv = requests.get(url_ctv)
print(r_ctv.text)


data_path_seg = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
j_data_pathseg = json.dumps(data_path_seg)

r_pathseg = requests.post(url_pathseg, data=j_data_pathseg, headers=headers)
pathseg_result = r_pathseg.text
print(pathseg_result)
