# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:25:28 2021

@author: mjichkar
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd
import re
from json_struc import * 

app = Flask(__name__)

@app.route('/api/coxph', methods=['POST'])
def coxph():
    data = request.get_json()
   
    cols = ['age', 'alcohol_consumption', 'amenorrhea', 'anxiety', 'bc_implant', 'bc_injection', 
        'bc_oral', 'bc_patch', 'bi_oophorectomy', 'bloating', 'depression', 'dizziness', 'dry_skin',
        'dyspareunia', 'endometrial ablation', 'fatigue', 'hair_loss', 'headache_migraine_rx', 'hot_flash_sev', 
        'hrt', 'hysterectomy', 'irritability', 'memory_lapse', 'menstrual_changes', 'night_sweats_rx', 
        'oab_incontinence', 'oligomenorrhea', 'sexual_dysfunction', 'sleep_disturbance', 'smoker', 
        'stress_incontinence', 'uni_oophorectomy', 'urge_incontinence', 'uti', 'vaginal_dryness_rx', 
        'weight_gain', 'bmi', 'race_ASIAN', 'race_CAUCASIAN', 'race_HISPANIC', 'race_OTHER', 'race_UNKNOWN']
    data = pd.DataFrame(data,columns=cols)
    
    pred_cum_haz = str(model_coxph.predict_cumulative_hazard(data))
    pred_sur     = str(model_coxph.predict_survival_function(data))
   
    json_data = coxph_json(pred_cum_haz,pred_sur)
    
    return json_data

@app.route('/api/aam', methods=['POST'])
def aam():
    data = request.get_json()
    #print(data)
    cols = ['age', 'alcohol_consumption', 'amenorrhea', 'anxiety', 'bc_implant', 'bc_injection', 
        'bc_oral', 'bc_patch', 'bi_oophorectomy', 'bloating', 'depression', 'dizziness', 'dry_skin',
        'dyspareunia', 'endometrial ablation', 'fatigue', 'hair_loss', 'headache_migraine_rx', 'hot_flash_sev', 
        'hrt', 'hysterectomy', 'irritability', 'memory_lapse', 'menstrual_changes', 'night_sweats_rx', 
        'oab_incontinence', 'oligomenorrhea', 'sexual_dysfunction', 'sleep_disturbance', 'smoker', 
        'stress_incontinence', 'uni_oophorectomy', 'urge_incontinence', 'uti', 'vaginal_dryness_rx', 
        'weight_gain', 'bmi', 'race_ASIAN', 'race_CAUCASIAN', 'race_HISPANIC', 'race_OTHER', 'race_UNKNOWN']
    data = pd.DataFrame(data,columns=cols)

    pred_cum_haz = str(model_aam.predict_cumulative_hazard(data))
    pred_sur     = str(model_aam.predict_survival_function(data))
    
    json_data = coxph_json(pred_cum_haz,pred_sur)
    
    return json_data

@app.route('/api/ctv')
def ctv():
    prediction = model_ctv.summary
    return str(prediction)

@app.route('/api/pathseg', methods=['POST'])
def pathseg():
    data = request.get_json()
    
    data = np.array(data)
    pred = model_pathseg.predict(data)
    
    return jsonify(str(pred[0]))


if __name__ == '__main__':
    modelfile_coxph = '//usvaprdsas01/rwi/eas/sentier/cph.pickle'
    modelfile_aam = '//usvaprdsas01/rwi/eas/sentier/aaf.pickle'
    modelfile_ctv = '//usvaprdsas01/rwi/eas/sentier/ctv.pickle'
    modelfile_pathseg = '//usvaprdsas01/rwi/eas/sentier/pathseg.pickle'
    
    
    model_coxph = p.load(open(modelfile_coxph, 'rb'))
    model_aam = p.load(open(modelfile_aam, 'rb'))
    model_ctv = p.load(open(modelfile_ctv, 'rb'))
    model_pathseg = p.load(open(modelfile_pathseg, 'rb'))
    
    app.run(debug=False, host='0.0.0.0')
    
