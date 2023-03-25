# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:16:19 2021

@author: mjichkar
"""

import numpy as np
import json
import pandas as pd
import re

def coxph_aam_json(pred1,pred2):
    
    #temp = r_coxph.text
    temp = pred1
    temp = re.split(r'\\n|\n', temp)
    
    temp_df = pd.DataFrame(temp,columns=['A'])
    temp_df[['Age', 'Pred']] = temp_df['A'].str.split(' ', 1, expand=True)
    temp_df = temp_df[['Age', 'Pred']]
    temp_df = temp_df.drop(temp_df.index[0]).drop(temp_df.index[len(temp_df)-1])
    
    temp_df['Pred'] = temp_df['Pred'].str.replace('"','')
    temp_df['Pred'] = temp_df['Pred'].str.strip()
    
    x = {}
    for i in range(len(temp_df)):
        x[temp_df.iloc[i,0]] = temp_df.iloc[i,1]
    
    master = {}
    master['Cummulative_Hazard'] = x
    
    temp = pred2
    temp = re.split(r'\\n|\n', temp)
    
    temp_df = pd.DataFrame(temp,columns=['A'])
    temp_df[['Age', 'Pred']] = temp_df['A'].str.split(' ', 1, expand=True)
    temp_df = temp_df[['Age', 'Pred']]
    temp_df = temp_df.drop(temp_df.index[0]).drop(temp_df.index[len(temp_df)-1])
    
    temp_df['Pred'] = temp_df['Pred'].str.replace('"','')
    temp_df['Pred'] = temp_df['Pred'].str.strip()
    
    y = {}
    for i in range(len(temp_df)):
        y[temp_df.iloc[i,0]] = temp_df.iloc[i,1]

    master['Survival_Function'] = y
    
    json_data = json.dumps(master)
    json_data = str(json_data).replace('"','')
    return json_data