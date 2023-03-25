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
    cols = ['age', 'alcohol_consumption', 'amenorrhea', 'anxiety', 'bc_implant', 'bc_injection', 
        'bc_oral', 'bc_patch', 'bi_oophorectomy', 'bloating', 'depression', 'dizziness', 'dry_skin',
        'dyspareunia', 'endometrial ablation', 'fatigue', 'hair_loss', 'headache_migraine_rx', 'hot_flash_sev', 
        'hrt', 'hysterectomy', 'irritability', 'memory_lapse', 'menstrual_changes', 'night_sweats_rx', 
        'oab_incontinence', 'oligomenorrhea', 'sexual_dysfunction', 'sleep_disturbance', 'smoker', 
        'stress_incontinence', 'uni_oophorectomy', 'urge_incontinence', 'uti', 'vaginal_dryness_rx', 
        'weight_gain', 'bmi', 'race_ASIAN', 'race_CAUCASIAN', 'race_HISPANIC', 'race_OTHER', 'race_UNKNOWN']
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
    cols_list = ['age',
     'alcohol_consumption',
     'amenorrhea',
     'anxiety',
     'bc_implant',
     'bc_injection',
     'bc_oral',
     'bc_patch',
     'bi_oophorectomy',
     'bloating',
     'depression',
     'dizziness',
     'dry_skin',
     'dyspareunia',
     'endometrial ablation',
     'fatigue',
     'hair_loss',
     'headache_migraine_rx',
     'hot_flash_sev',
     'hrt',
     'hysterectomy',
     'irritability',
     'memory_lapse',
     'menstrual_changes',
     'night_sweats_rx',
     'oab_incontinence',
     'oligomenorrhea',
     'sexual_dysfunction',
     'sleep_disturbance',
     'smoker',
     'stress_incontinence',
     'uni_oophorectomy',
     'urge_incontinence',
     'uti',
     'vaginal_dryness_rx',
     'weight_gain',
     'bmi',
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
        cols_list = ['anxiety','depression','dizziness','fatigue','headache_migraine_rx','menstrual_changes','oab_incontinence','osteoporosis','sleep_disturbance','weight_gain']
        for i in range(41,61):
            if(age == i):
                temp_list.append(1)
            else:
                temp_list.append(0)
        del data['age']
    else:
        cols_list = ['anxiety','depression','dizziness','fatigue','headache_migraine_rx','oab_incontinence','osteoporosis','sleep_disturbance','vaginal_dryness_sev','weight_gain']
        for i in range(40,61):
            if(age == i):
                temp_list.append(1)
            else:
                temp_list.append(0)
        del data['age']
      
    for i in range(len(cols_list)):
        temp_list.append(data[cols_list[i]])
    data_list = [temp_list]
    return data_list

def get_path_seg_summary():
    data = ""
    for k in pathseg_summary_json:
        data += "<td>" + str(k) + "</td>"
        #print(k)
        for d in pathseg_summary_json[k]:
            #print(d)
            data += "<td>" + str(d) +":"+ str(pathseg_summary_json[k][d]) + "</td>"
        data += "<tr>"
          
    data = "<table border=1>" + data + "<table>"
    #print(data)
    with open("templates/index.html", "w") as file:
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
    'patient_id':fields.Float(description="Unique Patient Identifier", required=True),
    'age':fields.Float(description="Range: 40-60", required=True, default = 41),
    'race':fields.String(description="Asian/Caucasian/Hispanic/Other/Unknown", required=True),
    'alcohol_consumption':fields.Float(description="0/1", required=True), 
    'amenorrhea':fields.Float(description="0/1", required=True),
    'anxiety':fields.Float(description="0/1", required=True),
    'bc_implant':fields.Float(description="0/1", required=True),
    'bc_injection':fields.Float(description="0/1", required=True),
    'bc_oral':fields.Float(description="0/1", required=True),
    'bc_other':fields.Float(description="0/1", required=True),
    'bc_patch':fields.Float(description="0/1", required=True),
    'bi_oophorectomy':fields.Float(description="0/1", required=True),
    'birth_control':fields.Float(description="0/1", required=True),
    'bloating':fields.Float(description="0/1", required=True),
    'bmi':fields.Float(description="BMI Range", required=True),
    'breast_cancer':fields.Float(description="0/1", required=True),
    'cancer':fields.Float(description="0/1", required=True), 
    'dec_libido':fields.Float(description="0/1", required=True), 
    'depression':fields.Float(description="0/1", required=True),
    'dizziness':fields.Float(description="0/1", required=True), 
    'dry_skin':fields.Float(description="0/1", required=True), 
    'dyspareunia':fields.Float(description="0/1", required=True),
    'endometrial ablation':fields.Float(description="0/1", required=True), 
    'fatigue':fields.Float(description="0/1", required=True), 
    'hair_loss':fields.Float(description="0/1", required=True), 
    'headache_migraine':fields.Float(description="0/1", required=True), 
    'headache_migraine_freq':fields.Float(description="0/1", required=True), 
    'headache_migraine_rx':fields.Float(description="0/1", required=True), 
    'hot_flash':fields.Float(description="0/1", required=True),
    'hot_flash_freq':fields.Float(description="0/1", required=True), 
    'hot_flash_rx':fields.Float(description="0/1", required=True), 
    'hot_flash_sev':fields.Float(description="0/1", required=True), 
    'hrt':fields.Float(description="0/1", required=True), 
    'hyst_oophorectomy':fields.Float(description="0/1", required=True),
    'hysterectomy':fields.Float(description="0/1", required=True),
    'incontinence':fields.Float(description="0/1", required=True), 
    'irritability':fields.Float(description="0/1", required=True), 
    'last_period':fields.Float(description="0/1", required=True),
    'memory_lapse':fields.Float(description="0/1", required=True), 
    'menopause':fields.Float(description="0/1", required=True), 
    'menstrual_changes':fields.Float(description="0/1", required=True),
    'night_sweats':fields.Float(description="0/1", required=True),
    'night_sweats_freq':fields.Float(description="0/1", required=True),
    'night_sweats_rx':fields.Float(description="0/1", required=True), 
    'night_sweats_sev':fields.Float(description="0/1", required=True),
    'oab_incontinence':fields.Float(description="0/1", required=True), 
    'oligomenorrhea':fields.Float(description="0/1", required=True), 
    'osteoporosis':fields.Float(description="0/1", required=True), 
    'sexual_dysfunction':fields.Float(description="0/1", required=True),
    'sleep_disturbance':fields.Float(description="0/1", required=True),
    'smoker':fields.Float(description="0/1", required=True), 
    'stress_incontinence':fields.Float(description="0/1", required=True),
    'uni_oophorectomy':fields.Float(description="0/1", required=True), 
    'urge_incontinence':fields.Float(description="0/1", required=True), 
    'uti':fields.Float(description="0/1", required=True), 
    'vaginal_dryness':fields.Float(description="0/1", required=True),
    'vaginal_dryness_freq':fields.Float(description="0/1", required=True),
    'vaginal_dryness_rx':fields.Float(description="0/1", required=True),
    'vaginal_dryness_sev':fields.Float(description="0/1", required=True), 
    'weight_gain':fields.Float(description="0/1", required=True),
    'post_meno':fields.Float(description="0/1 \n 0:PreMeno 1:PostMeno", required=True),
}

pathseg_summary_json = {"Cluster 1":
{
'amenorrhea':	12.5
,'anxiety':	84.05
,'bloating':	36.36
,'depression':	84.21
,'dizziness':	51.14
,'dry_skin':	36.84
,'fatigue':	72.5
,'hair_loss':	27.27
,'headache_migraine_rx':	65.15
,'hot_flash_sev':	22.22
,'memory_lapse':	52.63
,'menstrual_changes':	70.91
,'night_sweats_sev':	50
,'oab_incontinence':	56.86
,'oligomenorrhea':	17.65
,'osteoporosis':	54.13
,'sexual_dysfunction':	45.45
,'sleep_disturbance':	81.25
,'vaginal_dryness_sev':	41.67
,'weight_gain':	81.1
},
"Cluster 2":    
{
'amenorrhea':	0
,'anxiety':	7.86
,'bloating':	0
,'depression':	5.47
,'dizziness':	25
,'dry_skin':	0
,'fatigue':	5.5
,'hair_loss':	9.09
,'headache_migraine_rx':	17.42
,'hot_flash_sev':	0
,'memory_lapse':	0
,'menstrual_changes':	1.82
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	5.88
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	3.37
,'vaginal_dryness_sev':	0
,'weight_gain':	5.51
},
"Cluster 3":
    {
'amenorrhea':	0
,'anxiety':	2.38
,'bloating':	0
,'depression':	2.53
,'dizziness':	3.41
,'dry_skin':	5.26
,'fatigue':	7.5
,'hair_loss':	4.55
,'headache_migraine_rx':	0.76
,'hot_flash_sev':	11.11
,'memory_lapse':	5.26
,'menstrual_changes':	5.45
,'night_sweats_sev':	0
,'oab_incontinence':	6.86
,'oligomenorrhea':	0
,'osteoporosis':	11.93
,'sexual_dysfunction':	0
,'sleep_disturbance':	3.37
,'vaginal_dryness_sev':	20.83
,'weight_gain':	4.72

     },
"Cluster 4":
    {
     'amenorrhea':	12.5
,'anxiety':	2.86
,'bloating':	6.06
,'depression':	4.21
,'dizziness':	4.55
,'dry_skin':	0
,'fatigue':	1.5
,'hair_loss':	0
,'headache_migraine_rx':	0.76
,'hot_flash_sev':	0
,'memory_lapse':	0
,'menstrual_changes':	7.27
,'night_sweats_sev':	10
,'oab_incontinence':	9.8
,'oligomenorrhea':	23.53
,'osteoporosis':	7.34
,'sexual_dysfunction':	0
,'sleep_disturbance':	4.81
,'vaginal_dryness_sev':	4.17
,'weight_gain':	3.15
},
    "Cluster 5":
        {
            'amenorrhea':	0
,'anxiety':	1.43
,'bloating':	0
,'depression':	2.11
,'dizziness':	4.55
,'dry_skin':	15.79
,'fatigue':	2
,'hair_loss':	0
,'headache_migraine_rx':	3.03
,'hot_flash_sev':	22.22
,'memory_lapse':	0
,'menstrual_changes':	1.82
,'night_sweats_sev':	0
,'oab_incontinence':	7.84
,'oligomenorrhea':	0
,'osteoporosis':	2.75
,'sexual_dysfunction':	0
,'sleep_disturbance':	2.88
,'vaginal_dryness_sev':	0
,'weight_gain':	5.51
},
    "Cluster 6":
        {
        'amenorrhea':	12.5
,'anxiety':	0
,'bloating':	3.03
,'depression':	0
,'dizziness':	5.68
,'dry_skin':	0
,'fatigue':	3.5
,'hair_loss':	0
,'headache_migraine_rx':	11.36
,'hot_flash_sev':	0
,'memory_lapse':	0
,'menstrual_changes':	9.09
,'night_sweats_sev':	10
,'oab_incontinence':	5.88
,'oligomenorrhea':	5.88
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	2.88
,'vaginal_dryness_sev':	0
,'weight_gain':	0
},
    "Cluster 7":
        {
        'amenorrhea':	0
,'anxiety':	1.43
,'bloating':	9.09
,'depression':	1.26
,'dizziness':	5.68
,'dry_skin':	5.26
,'fatigue':	1
,'hair_loss':	4.55
,'headache_migraine_rx':	1.52
,'hot_flash_sev':	11.11
,'memory_lapse':	0
,'menstrual_changes':	1.82
,'night_sweats_sev':	0
,'oab_incontinence':	12.75
,'oligomenorrhea':	0
,'osteoporosis':	9.17
,'sexual_dysfunction':	0
,'sleep_disturbance':	0.48
,'vaginal_dryness_sev':	4.17
,'weight_gain':	0
},
    "Cluster 8":
        {
        'amenorrhea':	0
,'anxiety':	0
,'bloating':	9.09
,'depression':	0
,'dizziness':	0
,'dry_skin':	5.26
,'fatigue':	0
,'hair_loss':	0
,'headache_migraine_rx':	0
,'hot_flash_sev':	0
,'memory_lapse':	0
,'menstrual_changes':	0
,'night_sweats_sev':	10
,'oab_incontinence':	0
,'oligomenorrhea':	0
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
},
    "Cluster 9":
        {
        'amenorrhea':	0
,'anxiety':	0
,'bloating':	0
,'depression':	0
,'dizziness':	0
,'dry_skin':	15.79
,'fatigue':	0
,'hair_loss':	4.55
,'headache_migraine_rx':	0
,'hot_flash_sev':	11.11
,'memory_lapse':	15.79
,'menstrual_changes':	0
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	17.65
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
  
},
        "Cluster 10":
        {
        'amenorrhea':	25
,'anxiety':	0
,'bloating':	0
,'depression':	0
,'dizziness':	0
,'dry_skin':	5.26
,'fatigue':	0
,'hair_loss':	0
,'headache_migraine_rx':	0
,'hot_flash_sev':	0
,'memory_lapse':	5.26
,'menstrual_changes':	0
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	11.76
,'osteoporosis':	0
,'sexual_dysfunction':	27.27
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
},
        "Cluster 11":
        {
        'amenorrhea':	12.5
,'anxiety':	0
,'bloating':	9.09
,'depression':	0
,'dizziness':	0
,'dry_skin':	0
,'fatigue':	0
,'hair_loss':	18.18
,'headache_migraine_rx':	0
,'hot_flash_sev':	0
,'memory_lapse':	10.53
,'menstrual_changes':	0
,'night_sweats_sev':	10
,'oab_incontinence':	0
,'oligomenorrhea':	0
,'osteoporosis':	0
,'sexual_dysfunction':	18.18
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
},
    "Cluster 12":
        {
        'amenorrhea':	0
,'anxiety':	0
,'bloating':	3.03
,'depression':	0
,'dizziness':	0
,'dry_skin':	5.26
,'fatigue':	0
,'hair_loss':	27.27
,'headache_migraine_rx':	0
,'hot_flash_sev':	11.11
,'memory_lapse':	0
,'menstrual_changes':	0
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	0
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	4.17
,'weight_gain':	0
},
        "Cluster 13":
        {
        'amenorrhea':	0
,'anxiety':	0
,'bloating':	12.12
,'depression':	0.21
,'dizziness':	0
,'dry_skin':	0
,'fatigue':	6.5
,'hair_loss':	4.55
,'headache_migraine_rx':	0
,'hot_flash_sev':	0
,'memory_lapse':	10.53
,'menstrual_changes':	1.82
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	0
,'osteoporosis':	14.68
,'sexual_dysfunction':	0
,'sleep_disturbance':	0.96
,'vaginal_dryness_sev':	25
,'weight_gain':	0
},
    "Cluster 14":
    {
     'amenorrhea':	12.5
,'anxiety':	0
,'bloating':	6.06
,'depression':	0
,'dizziness':	0
,'dry_skin':	0
,'fatigue':	0
,'hair_loss':	0
,'headache_migraine_rx':	0
,'hot_flash_sev':	0
,'memory_lapse':	0
,'menstrual_changes':	0
,'night_sweats_sev':	0
,'oab_incontinence':	0
,'oligomenorrhea':	0
,'osteoporosis':	0
,'sexual_dysfunction':	0
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
},
    "Cluster 15":
    {
     'amenorrhea':	12.5
,'anxiety':	0
,'bloating':	6.06
,'depression':	0
,'dizziness':	0
,'dry_skin':	5.26
,'fatigue':	0
,'hair_loss':	0
,'headache_migraine_rx':	0
,'hot_flash_sev':	11.11
,'memory_lapse':	0
,'menstrual_changes':	0
,'night_sweats_sev':	10
,'oab_incontinence':	0
,'oligomenorrhea':	17.65
,'osteoporosis':	0
,'sexual_dysfunction':	9.09
,'sleep_disturbance':	0
,'vaginal_dryness_sev':	0
,'weight_gain':	0
}
}

'''
pathseg_summary_json = {"Cluster 1":
{amenorrhea:	1,
anxiety:	353,
bloating:	12,
depression:	400,
dizziness:	45,
dry_skin:	7,
fatigue:	145,
hair_loss:	6,
headache_migraine_rx:	86,
hot_flash_sev:	2,
memory_lapse:	10,
menstrual_changes:	39,
night_sweats_sev:	5,
oab_incontinence:	58,
oligomenorrhea:	3,
osteoporosis:	59,
sexual_dysfunction:	5,
sleep_disturbance:	169,
vaginal_dryness_sev:	10,
weight_gain:	309,
},
"Cluster 2":
{
amenorrhea:	0,
anxiety:	33,
bloating:	0,
depression:	26,
dizziness:	22,
dry_skin:	0,
fatigue:	11,
hair_loss:	2,
headache_migraine_rx:	23,
hot_flash_sev:	0,
memory_lapse:	0,
menstrual_changes:	1,
night_sweats_sev:	0,
oab_incontinence:	0,
oligomenorrhea:	1,
osteoporosis:	0,
sexual_dysfunction:	0,
sleep_disturbance:	7,
vaginal_dryness_sev:	0,
weight_gain:	21,

}
}
'''
