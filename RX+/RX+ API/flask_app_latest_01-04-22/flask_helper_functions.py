# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:28:42 2021

@author: mjichkar
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd
import re
from json_struc import * 
from flask_restplus import Api, Resource, fields 

def cox_aam(data,model):  
    cols = ['age', 'alcohol_consumption', 'amenorrhea', 'bc_implant', 'bc_injection', 'bc_oral', 
            'bi_oophorectomy', 'dec_libido', 'dry_skin', 'dyspareunia', 'fatigue', 'hair_loss', 
            'headache_migraine', 'hot_flash', 'hot_flash_rx', 'hysterectomy', 'last_period', 
            'memory_lapse', 'menopause', 'night_sweats', 'night_sweats_rx', 'oab_incontinence', 
            'sleep_disturbance', 'smoker', 'stress_incontinence', 'uni_oophorectomy', 
            'urge_incontinence', 'uti', 'vaginal_dryness', 'weight_gain', 'race_ASIAN', 
            'race_CAUCASIAN', 'race_HISPANIC', 'race_OTHER', 'race_UNKNOWN']
    data = pd.DataFrame(data,columns=cols)
    
    pred_cum_haz = str(model.predict_cumulative_hazard(data))
    pred_sur     = str(model.predict_survival_function(data))
   
    json_data = coxph_aam_json(pred_cum_haz,pred_sur)
    return json_data

        
def get_ctv_summary(model_ctv):
    summary = model_ctv.summary
    return str(summary)

def get_pathseg_summary():
    return str(pathseg_summary_json)
    
def process_user_data(data):
    temp_list = []
    for i in data.keys():
        temp_list.append(data[i])
    data_list = [temp_list]
    return data_list
        

def process_user_data1(data):
    cols_list = ['age', 'alcohol_consumption', 'amenorrhea', 'bc_implant', 'bc_injection', 'bc_oral', 
                 'bi_oophorectomy', 'dec_libido', 'dry_skin', 'dyspareunia', 'fatigue', 
                 'hair_loss', 'headache_migraine', 'hot_flash', 'hot_flash_rx', 'hysterectomy',
                 'last_period', 'memory_lapse', 'menopause', 'night_sweats', 'night_sweats_rx', 
                 'oab_incontinence', 'sleep_disturbance', 'smoker', 'stress_incontinence', 
                 'uni_oophorectomy', 'urge_incontinence', 'uti', 'vaginal_dryness', 'weight_gain', 
                 'race']
    temp_list = []
    col_len = len(cols_list)
    for i in range(col_len-1):
        temp_list.append(data[cols_list[i]])
    if(data[cols_list[col_len-1]].lower()=='asian'):
        temp_list.append(1)
    else:
        temp_list.append(0)
    if(data[cols_list[col_len-1]].lower()=='caucasian'):
        temp_list.append(1)
    else:
        temp_list.append(0)  
    if(data[cols_list[col_len-1]].lower()=='hispanic'):
        temp_list.append(1)
    else:
        temp_list.append(0)
    if(data[cols_list[col_len-1]].lower()=='other'):
        temp_list.append(1)
    else:
        temp_list.append(0)  
    if(data[cols_list[col_len-1]].lower()=='unknown'):
        temp_list.append(1)
    else:
        temp_list.append(0)
    data_list = [temp_list]
    return data_list     

def process_user_data_segmentation(data):
    temp_list = []
    age = data['age']
    if(data['post_meno']==0):
        cols_list = ['alcohol_consumption', 'bc_oral', 'birth_control', 'fatigue', 'headache_migraine', 'oab_incontinence', 'sleep_disturbance', 'uti', 'weight_gain']
        for i in range(41,60): 
            if(age == i):
                temp_list.append(1)
            else:
                temp_list.append(0)
        del data['age']
    else:
        cols_list = ['alcohol_consumption', 'bc_oral', 'birth_control','fatigue','headache_migraine','hot_flash','last_period','menopause','oab_incontinence','sleep_disturbance','uti','vaginal_dryness','weight_gain']
        for i in range(41,61):
            if(age == i):
                temp_list.append(1)
            else:
                temp_list.append(0)
        del data['age']
    for i in range(len(cols_list)):
        temp_list.append(data[cols_list[i]])
    data_list = [temp_list]
    return data_list

def get_path_seg_summary_pre():
    data = ""
    for k in pathseg_summary_json_pre:
        data += "<td>" + str(k) + "</td>"
        #print(k)
        for d in pathseg_summary_json_pre[k]:
            #print(d)
            data += "<td>" + str(d) +":"+ str(pathseg_summary_json_pre[k][d]) + "</td>"
        data += "<tr>"
          
    data = "<table border=1>" + data + "<table>"
    with open("templates/index_pre.html", "w") as file:
        file.write(data)
    # return render_template('index.html', title="page", jsonfile=json.dumps(data))
    return data

def get_path_seg_summary_post():
    data = ""
    for k in pathseg_summary_json_post:
        data += "<td>" + str(k) + "</td>"
        #print(k)
        for d in pathseg_summary_json_post[k]:
            #print(d)
            data += "<td>" + str(d) +":"+ str(pathseg_summary_json_post[k][d]) + "</td>"
        data += "<tr>"
    
    data = "<table border=1>" + data + "<table>"
    with open("templates/index_post.html", "w") as file:
        file.write(data)
    # return render_template('index.html', title="page", jsonfile=json.dumps(data))
    return data

user_data_format = {
 'age':fields.Float(description="", required=True,default = 41),
 'alcohol_consumption':fields.Float(description="", required=True),
 'amenorrhea':fields.Float(description="", required=True),
 'anxiety':fields.Float(description="", required=True),
 'bc_implant':fields.Float(description="", required=True),
 'bc_injection':fields.Float(description="", required=True),
 'bc_oral':fields.Float(description="", required=True),
 'bc_patch':fields.Float(description="", required=True),
 'bi_oophorectomy':fields.Float(description="", required=True),
 'bloating':fields.Float(description="", required=True),
 'depression':fields.Float(description="", required=True),
 'dizziness':fields.Float(description="", required=True),
 'dry_skin':fields.Float(description="", required=True),
 'dyspareunia':fields.Float(description="", required=True),
 'endometrial ablation':fields.Float(description="", required=True),
 'fatigue':fields.Float(description="", required=True),
 'hair_loss':fields.Float(description="", required=True),
 'headache_migraine_rx':fields.Float(description="", required=True),
 'hot_flash_sev':fields.Float(description="", required=True),
 'hrt':fields.Float(description="", required=True),
 'hysterectomy':fields.Float(description="", required=True),
 'irritability':fields.Float(description="", required=True),
 'memory_lapse':fields.Float(description="", required=True),
 'menstrual_changes':fields.Float(description="", required=True),
 'night_sweats_rx':fields.Float(description="", required=True),
 'oab_incontinence':fields.Float(description="", required=True),
 'oligomenorrhea':fields.Float(description="", required=True),
 'sexual_dysfunction':fields.Float(description="", required=True),
 'sleep_disturbance':fields.Float(description="", required=True),
 'smoker':fields.Float(description="", required=True),
 'stress_incontinence':fields.Float(description="", required=True),
 'uni_oophorectomy':fields.Float(description="", required=True),
 'urge_incontinence':fields.Float(description="", required=True),
 'uti':fields.Float(description="", required=True),
 'vaginal_dryness_rx':fields.Float(description="", required=True),
 'weight_gain':fields.Float(description="", required=True),
 'bmi':fields.Float(description="", required=True),
 'race_ASIAN':fields.Float(description="", required=True),
 'race_CAUCASIAN':fields.Float(description="", required=True),
 'race_HISPANIC':fields.Float(description="", required=True),
 'race_OTHER':fields.Float(description="", required=True),
 'race_UNKNOWN':fields.Float(description="", required=True)}


user_data_format_segmentation = {
    'patient_id':fields.Float(description="Unique Patient Identifier: \n", required=True),
    'age':fields.Float(description="Range: 40-60", required=True, default = 41),
    'race':fields.String(description="Asian/Caucasian/Hispanic/Other/Unknown"),
    'alcohol_consumption':fields.Float(description="0/1", required=True),  
    'amenorrhea':fields.Float(description="0/1", required=True),
    'anxiety':fields.Float(description="0/1"),
    'bc_implant':fields.Float(description="0/1", required=True),
    'bc_injection':fields.Float(description="0/1", required=True),
    'bc_oral':fields.Float(description="0/1", required=True),
    'bc_other':fields.Float(description="0/1", required=True),
    'bc_patch':fields.Float(description="0/1"),
    'bi_oophorectomy':fields.Float(description="0/1", required=True),
    'birth_control':fields.Float(description="0/1", required=True),
    'bloating':fields.Float(description="0/1"),
    'bmi':fields.Float(description="BMI Range: \n"),
    'breast_cancer':fields.Float(description="0/1"),
    'cancer':fields.Float(description="0/1"), 
    'dec_libido':fields.Float(description="0/1", required=True), 
    'depression':fields.Float(description="0/1"),
    'dizziness':fields.Float(description="0/1"), 
    'dry_skin':fields.Float(description="0/1", required=True), 
    'dyspareunia':fields.Float(description="0/1", required=True),
    'endometrial ablation':fields.Float(description="0/1"), 
    'fatigue':fields.Float(description="0/1", required=True), 
    'hair_loss':fields.Float(description="0/1", required=True), 
    'headache_migraine':fields.Float(description="0/1", required=True), 
    'headache_migraine_freq':fields.Float(description="0/1"), 
    'headache_migraine_rx':fields.Float(description="0/1"), 
    'hot_flash':fields.Float(description="0/1", required=True),
    'hot_flash_freq':fields.Float(description="0/1"), 
    'hot_flash_rx':fields.Float(description="0/1", required=True), 
    'hot_flash_sev':fields.Float(description="0/1"), 
    'hrt':fields.Float(description="0/1"), 
    'hyst_oophorectomy':fields.Float(description="0/1", required=True),
    'hysterectomy':fields.Float(description="0/1", required=True),
    'incontinence':fields.Float(description="0/1"), 
    'irritability':fields.Float(description="0/1"), 
    'last_period':fields.Float(description="0/1", required=True),
    'memory_lapse':fields.Float(description="0/1", required=True), 
    'menopause':fields.Float(description="0/1", required=True), 
    'menstrual_changes':fields.Float(description="0/1"),
    'night_sweats':fields.Float(description="0/1", required=True),
    'night_sweats_freq':fields.Float(description="0/1"),
    'night_sweats_rx':fields.Float(description="0/1", required=True), 
    'night_sweats_sev':fields.Float(description="0/1"),
    'oab_incontinence':fields.Float(description="0/1", required=True), 
    'oligomenorrhea':fields.Float(description="0/1"), 
    'osteoporosis':fields.Float(description="0/1"), 
    'sexual_dysfunction':fields.Float(description="0/1"),
    'sleep_disturbance':fields.Float(description="0/1", required=True),
    'smoker':fields.Float(description="0/1", required=True), 
    'stress_incontinence':fields.Float(description="0/1", required=True),
    'uni_oophorectomy':fields.Float(description="0/1", required=True), 
    'urge_incontinence':fields.Float(description="0/1", required=True), 
    'uti':fields.Float(description="0/1", required=True), 
    'vaginal_dryness':fields.Float(description="0/1", required=True),
    'vaginal_dryness_freq':fields.Float(description="0/1"),
    'vaginal_dryness_rx':fields.Float(description="0/1"),
    'vaginal_dryness_sev':fields.Float(description="0/1"), 
    'weight_gain':fields.Float(description="0/1", required=True),
    'post_meno':fields.Float(description="0/1 \n 0:PreMeno 1:PostMeno \n Not significant for Survival Analysis Models", required=True),
}


pathseg_summary_json_pre = {"Cluster 1":
{
'alcohol_consumption':64.17
,'amenorrhea':25.81
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':18.18
,'dyspareunia':0
,'fatigue':42.22
,'hair_loss':34.78
,'headache_migraine':45.71
,'hot_flash':14.29
,'hot_flash_rx':50
,'hyst_oophorectomy':33.33
,'hysterectomy':0
,'last_period':0
,'memory_lapse':14.29
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':0
,'oab_incontinence':49.29
,'sleep_disturbance':61.54
,'smoker':5
,'stress_incontinence':38.78
,'uni_oophorectomy':0
,'urge_incontinence':38.89
,'uti':37.24
,'vaginal_dryness':0
,'weight_gain':48.41
},
"Cluster 2":    
{
'alcohol_consumption':3.33
,'amenorrhea':22.58
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':9.09
,'dyspareunia':0
,'fatigue':8.15
,'hair_loss':0
,'headache_migraine':10.8
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':9.52
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':0
,'oab_incontinence':11.43
,'sleep_disturbance':5.86
,'smoker':35
,'stress_incontinence':18.37
,'uni_oophorectomy':0
,'urge_incontinence':19.44
,'uti':13.79
,'vaginal_dryness':0
,'weight_gain':11.64

},
"Cluster 3":
    {
'alcohol_consumption':0
,'amenorrhea':3.23
,'bc_implant':75
,'bc_injection':100
,'bc_oral':100
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':98.58
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':1.11
,'hair_loss':0
,'headache_migraine':0.83
,'hot_flash':7.14
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0.37
,'smoker':5
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0.69
,'vaginal_dryness':0
,'weight_gain':0.26


     },
"Cluster 4":
    {
'alcohol_consumption':0
,'amenorrhea':22.58
,'bc_implant':25
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':1.42
,'dec_libido':0
,'dry_skin':15.15
,'dyspareunia':0
,'fatigue':14.44
,'hair_loss':8.7
,'headache_migraine':6.65
,'hot_flash':7.14
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':23.81
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':0
,'oab_incontinence':5
,'sleep_disturbance':0
,'smoker':10
,'stress_incontinence':2.04
,'uni_oophorectomy':0
,'urge_incontinence':2.78
,'uti':6.21
,'vaginal_dryness':0
,'weight_gain':5.56

},
    "Cluster 5":
        {
'alcohol_consumption':7.5
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':40
,'dry_skin':12.12
,'dyspareunia':0
,'fatigue':11.11
,'hair_loss':13.04
,'headache_migraine':10.8
,'hot_flash':14.29
,'hot_flash_rx':0
,'hyst_oophorectomy':33.33
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':18.18
,'night_sweats_rx':0
,'oab_incontinence':6.43
,'sleep_disturbance':7.33
,'smoker':20
,'stress_incontinence':8.16
,'uni_oophorectomy':0
,'urge_incontinence':8.33
,'uti':15.86
,'vaginal_dryness':0
,'weight_gain':7.67

},
    "Cluster 6":
        {
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':20
,'dry_skin':12.12
,'dyspareunia':100
,'fatigue':0
,'hair_loss':8.7
,'headache_migraine':0
,'hot_flash':28.57
,'hot_flash_rx':50
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':10
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':50
,'weight_gain':0

},
    "Cluster 7":
        {
'alcohol_consumption':9.17
,'amenorrhea':6.45
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':10
,'hair_loss':0
,'headache_migraine':12.19
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':4.76
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':11.43
,'sleep_disturbance':12.09
,'smoker':0
,'stress_incontinence':16.33
,'uni_oophorectomy':0
,'urge_incontinence':13.89
,'uti':7.59
,'vaginal_dryness':0
,'weight_gain':15.61

},
    "Cluster 8":
        {
 'alcohol_consumption':15.83
,'amenorrhea':3.23
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':12.96
,'hair_loss':0
,'headache_migraine':13.02
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':9.52
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':50
,'oab_incontinence':16.43
,'sleep_disturbance':12.82
,'smoker':0
,'stress_incontinence':16.33
,'uni_oophorectomy':0
,'urge_incontinence':16.67
,'uti':18.62
,'vaginal_dryness':0
,'weight_gain':10.85

},
    "Cluster 9":
        {
  'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':20
,'dry_skin':6.06
,'dyspareunia':0
,'fatigue':0
,'hair_loss':4.35
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
        "Cluster 10":
        {
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':9.09
,'dyspareunia':0
,'fatigue':0
,'hair_loss':0
,'headache_migraine':0
,'hot_flash':7.14
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':4.76
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
        "Cluster 11":
        {
'alcohol_consumption':0
,'amenorrhea':6.45
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':12.12
,'dyspareunia':0
,'fatigue':0
,'hair_loss':17.39
,'headache_migraine':0
,'hot_flash':21.43
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':9.52
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':10
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
    "Cluster 12":
        {
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':0
,'hair_loss':0
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':33.33
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':5
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
        "Cluster 13":
        {
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':0
,'hair_loss':0
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':19.05
,'menopause':0
,'night_sweats':18.18
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
    "Cluster 14":
    {
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':20
,'dry_skin':6.06
,'dyspareunia':0
,'fatigue':0
,'hair_loss':4.35
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':4.76
,'menopause':0
,'night_sweats':9.09
,'night_sweats_rx':50
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':50
,'weight_gain':0

},
    "Cluster 15":
    {
'alcohol_consumption':0
,'amenorrhea':9.68
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':0
,'hair_loss':8.7
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

}
}

pathseg_summary_json_post = {"Cluster 1":
{
 'alcohol_consumption':47.18
,'amenorrhea':0
,'bc_implant':4.17
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0.13
,'dec_libido':6.38
,'dry_skin':16.9
,'dyspareunia':0
,'fatigue':45
,'hair_loss':20.97
,'headache_migraine':41.56
,'hot_flash':8.23
,'hot_flash_rx':3.77
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':53.2
,'memory_lapse':11.11
,'menopause':0
,'night_sweats':34.48
,'night_sweats_rx':25
,'oab_incontinence':44.05
,'sleep_disturbance':41.76
,'smoker':24.29
,'stress_incontinence':40.66
,'uni_oophorectomy':0
,'urge_incontinence':38.89
,'uti':38.91
,'vaginal_dryness':1.52
,'weight_gain':44.42

},
"Cluster 2":
{
 'alcohol_consumption':12.9
,'amenorrhea':0
,'bc_implant':12.5
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0.4
,'dec_libido':8.51
,'dry_skin':2.82
,'dyspareunia':0
,'fatigue':10.91
,'hair_loss':0
,'headache_migraine':12.76
,'hot_flash':9.49
,'hot_flash_rx':7.55
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':10.2
,'memory_lapse':5.56
,'menopause':12.71
,'night_sweats':3.45
,'night_sweats_rx':0
,'oab_incontinence':9.52
,'sleep_disturbance':14.9
,'smoker':0
,'stress_incontinence':9.89
,'uni_oophorectomy':0
,'urge_incontinence':13.89
,'uti':13.62
,'vaginal_dryness':16.75
,'weight_gain':13.02

},
"Cluster 3":
{
'alcohol_consumption':5.65
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':8.51
,'dry_skin':15.49
,'dyspareunia':0
,'fatigue':9.55
,'hair_loss':4.84
,'headache_migraine':8.02
,'hot_flash':8.86
,'hot_flash_rx':13.21
,'hyst_oophorectomy':28.57
,'hysterectomy':0
,'last_period':9.9
,'memory_lapse':5.56
,'menopause':10.08
,'night_sweats':3.45
,'night_sweats_rx':0
,'oab_incontinence':7.54
,'sleep_disturbance':8.04
,'smoker':2.86
,'stress_incontinence':6.59
,'uni_oophorectomy':0
,'urge_incontinence':13.89
,'uti':15.18
,'vaginal_dryness':11.68
,'weight_gain':10.07

},
"Cluster 4":
{
 'alcohol_consumption':7.66
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':4.26
,'dry_skin':9.86
,'dyspareunia':0
,'fatigue':4.32
,'hair_loss':4.84
,'headache_migraine':10.49
,'hot_flash':1.27
,'hot_flash_rx':0
,'hyst_oophorectomy':14.29
,'hysterectomy':0
,'last_period':8.4
,'memory_lapse':0
,'menopause':0
,'night_sweats':6.9
,'night_sweats_rx':25
,'oab_incontinence':10.32
,'sleep_disturbance':8.63
,'smoker':14.29
,'stress_incontinence':8.79
,'uni_oophorectomy':0
,'urge_incontinence':5.56
,'uti':10.12
,'vaginal_dryness':0
,'weight_gain':5.25

},
"Cluster 5":
{
 'alcohol_consumption':9.68
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':17.02
,'dry_skin':8.45
,'dyspareunia':0
,'fatigue':6.82
,'hair_loss':4.84
,'headache_migraine':9.88
,'hot_flash':1.27
,'hot_flash_rx':1.89
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':8.8
,'memory_lapse':5.56
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':10.32
,'sleep_disturbance':9.61
,'smoker':1.43
,'stress_incontinence':8.79
,'uni_oophorectomy':0
,'urge_incontinence':13.89
,'uti':6.61
,'vaginal_dryness':0
,'weight_gain':7.11

},
"Cluster 6":
{
 'alcohol_consumption':6.45
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':10.64
,'dry_skin':5.63
,'dyspareunia':50
,'fatigue':6.59
,'hair_loss':3.23
,'headache_migraine':4.94
,'hot_flash':1.27
,'hot_flash_rx':1.89
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':9.5
,'memory_lapse':5.56
,'menopause':0
,'night_sweats':3.45
,'night_sweats_rx':0
,'oab_incontinence':4.76
,'sleep_disturbance':6.67
,'smoker':5.71
,'stress_incontinence':4.4
,'uni_oophorectomy':0
,'urge_incontinence':2.78
,'uti':10.12
,'vaginal_dryness':3.05
,'weight_gain':8.42

},
"Cluster 7":
{
 'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':6.38
,'dry_skin':8.45
,'dyspareunia':0
,'fatigue':0
,'hair_loss':0
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':28.57
,'hysterectomy':0
,'last_period':0
,'memory_lapse':33.33
,'menopause':0
,'night_sweats':10.34
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':14.29
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
"Cluster 8":
{
 'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':1.41
,'dyspareunia':0
,'fatigue':0
,'hair_loss':1.61
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':5.56
,'menopause':0
,'night_sweats':10.34
,'night_sweats_rx':25
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':5.71
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
"Cluster 9":
{
 'alcohol_consumption':10.48
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':1.39
,'bc_oral':0.15
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0.27
,'dec_libido':29.79
,'dry_skin':15.49
,'dyspareunia':50
,'fatigue':16.59
,'hair_loss':27.42
,'headache_migraine':11.32
,'hot_flash':69.62
,'hot_flash_rx':71.7
,'hyst_oophorectomy':14.29
,'hysterectomy':0
,'last_period':0
,'memory_lapse':5.56
,'menopause':76.88
,'night_sweats':27.59
,'night_sweats_rx':25
,'oab_incontinence':13.49
,'sleep_disturbance':10.39
,'smoker':4.29
,'stress_incontinence':20.88
,'uni_oophorectomy':0
,'urge_incontinence':11.11
,'uti':5.06
,'vaginal_dryness':67.01
,'weight_gain':11.27

},
"Cluster 10":
{
 'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':5.63
,'dyspareunia':0
,'fatigue':0
,'hair_loss':14.52
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':14.29
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':10
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
"Cluster 11":
{
 'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':2.13
,'dry_skin':7.04
,'dyspareunia':0
,'fatigue':0
,'hair_loss':1.61
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

},
"Cluster 12":
{
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':0
,'dyspareunia':0
,'fatigue':0
,'hair_loss':4.84
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':16.67
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':2.86
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

 },
"Cluster 13":
{
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':83.33
,'bc_injection':98.61
,'bc_oral':99.85
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':99.2
,'dec_libido':6.38
,'dry_skin':0
,'dyspareunia':0
,'fatigue':0.23
,'hair_loss':0
,'headache_migraine':1.03
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0.33
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':0
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0.39
,'vaginal_dryness':0
,'weight_gain':0.44

 },
"Cluster 14":
{
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':1.41
,'dyspareunia':0
,'fatigue':0
,'hair_loss':3.23
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':0
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':4.29
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

 },
"Cluster 15":
{
'alcohol_consumption':0
,'amenorrhea':0
,'bc_implant':0
,'bc_injection':0
,'bc_oral':0
,'bc_other':0
,'bi_oophorectomy':0
,'birth_control':0
,'dec_libido':0
,'dry_skin':1.41
,'dyspareunia':0
,'fatigue':0
,'hair_loss':8.06
,'headache_migraine':0
,'hot_flash':0
,'hot_flash_rx':0
,'hyst_oophorectomy':0
,'hysterectomy':0
,'last_period':0
,'memory_lapse':5.56
,'menopause':0
,'night_sweats':0
,'night_sweats_rx':0
,'oab_incontinence':0
,'sleep_disturbance':0
,'smoker':10
,'stress_incontinence':0
,'uni_oophorectomy':0
,'urge_incontinence':0
,'uti':0
,'vaginal_dryness':0
,'weight_gain':0

 }
}


















 

