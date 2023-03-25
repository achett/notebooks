# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:10:11 2021

@author: mjichkar
"""


from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd

modelfile = '//usvaprdsas01/rwi/eas/sentier/final_prediction.pickle'
model = p.load(open(modelfile, 'rb'))

cols = ['age', 'alcohol_consumption', 'amenorrhea', 'anxiety', 'bc_implant', 'bc_injection', 
        'bc_oral', 'bc_patch', 'bi_oophorectomy', 'bloating', 'depression', 'dizziness', 'dry_skin',
        'dyspareunia', 'endometrial ablation', 'fatigue', 'hair_loss', 'headache_migraine_rx', 'hot_flash_sev', 
        'hrt', 'hysterectomy', 'irritability', 'memory_lapse', 'menstrual_changes', 'night_sweats_rx', 
        'oab_incontinence', 'oligomenorrhea', 'sexual_dysfunction', 'sleep_disturbance', 'smoker', 
        'stress_incontinence', 'uni_oophorectomy', 'urge_incontinence', 'uti', 'vaginal_dryness_rx', 
        'weight_gain', 'bmi', 'race_ASIAN', 'race_CAUCASIAN', 'race_HISPANIC', 'race_OTHER', 'race_UNKNOWN']

data = [[51,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29.32,0,1,0,0,0]]
data = pd.DataFrame(data,columns=cols)

prediction = model.predict_cumulative_hazard(data)
jsonify(prediction)

modelfile_ctv = '//usvaprdsas01/rwi/eas/sentier/ctv.pickle'
    
model_ctv = p.load(open(modelfile_ctv, 'rb'))
str(model_ctv.plot())
