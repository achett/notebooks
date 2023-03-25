# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:34:41 2021

@author: mjichkar
"""

#pip install flask_restplus
#pip install Werkzeug==0.16.1
#pip install flask_restful
#pip install pickle
#pip install lifelines

from flask_restplus import Api, Resource, fields 
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, make_response
from flask_restful import reqparse
import pickle as p
from flask_helper_functions import *
import os 

app = Flask(__name__)
Api = Api(app = app, 
		  version = "1.0", 
		  title = "Rx+ Models", 
		  description = "Survival Analysis and Path Segmentation related endpoints")

name_space1 = Api.namespace('Cox_PH_Model', description='Survival Analysis Cox Proportional Model')
user_data = name_space1.model("User_data",user_data_format_segmentation,)

@name_space1.route('/api/coxph')
class cox(Resource): 
    @name_space1.expect(user_data)           
    def post(self):
        data = process_user_data1(request.json)
        json_data = cox_aam(data,model_coxph)
        return json_data
    
name_space2 = Api.namespace('AAM_Model', description="Survival Analysis Aalen's Additive Model")
user_data = name_space2.model("User_data",user_data_format_segmentation,)

@name_space2.route('/api/aam')
class aam(Resource): 
    @name_space1.expect(user_data)           
    def post(self):
        data = process_user_data1(request.json)
        json_data = cox_aam(data,model_aam)
        return json_data


name_space = Api.namespace('Time_Varying_Model', description='Survival Analysis Time Varying Model')

@name_space.route('/api/ctv')
class ctv(Resource):
	def post(self):
		return {
			"status": "This is a static Model. Use 'get' method to get Model Summary."
		}
	def get(self):
		return get_ctv_summary(model_ctv)

name_space3 = Api.namespace('Path_Segmentation_Model', description="Path Segmentation Model")
#user_data_segmentation = name_space3.model("User_data_segmentation",user_data_format_segmentation,)
user_data = name_space3.model("User_data",user_data_format_segmentation,)

@name_space3.route('/api/pathseg')
class pathseg(Resource):
    #def get(self):
    #    return get_pathseg_summary()    
    @name_space1.expect(user_data)           
    def post(self):
        data = process_user_data_segmentation(request.json)
        data = np.array(data)
        #print(data)
        if(request.json['post_meno']==0):
            pred = model_pathseg.predict(data)
        else:
            pred = model_pathseg_post.predict(data)
        #print(*pred)
        return ('Cluster-'+str(*pred))

@name_space3.route('/api/pre-meno_summary')
class path_seg_summary_pre(Resource):
    def get(self):
        x = get_path_seg_summary_pre()
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index_pre.html', title="page"),200,headers)

@name_space3.route('/api/post-meno_summary')
class path_seg_summary_post(Resource):
    def get(self):
        x = get_path_seg_summary_post()
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index_post.html', title="page"),200,headers)
    
name_space4 = Api.namespace('RX+_Architecture&Flow', description="Architecture and Data Preparation Flow Details")

img_folder = os.path.join('\static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

@name_space4.route('/api/architecture')
class arch(Resource):
    def get(self):
        full_filename = full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Flask RX+ Architecture.png')
        return make_response(render_template('index_arch.html', user_image = full_filename),200)
    
@name_space4.route('/api/path_seg_flow')
class path_segmentation_flow(Resource):
    def get(self):
        full_filename = full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Path_Seg_Flow.jpg')
        return make_response(render_template('index_arch.html', user_image = full_filename),200)
 

if __name__ == '__main__':
    # modelfile_coxph = '//usvaprdsas01/rwi/eas/sentier/cph.pickle'
    # modelfile_aam = '//usvaprdsas01/rwi/eas/sentier/aaf.pickle'
    # modelfile_ctv = '//usvaprdsas01/rwi/eas/sentier/ctv.pickle'
    # modelfile_pathseg = '//usvaprdsas01/rwi/eas/sentier/pathseg.pickle'
    # modelfile_pathseg_post = '//usvaprdsas01/rwi/eas/sentier/pathseg_postmeno.pickle'
    
    modelfile_coxph = 'cph.pickle'
    modelfile_aam = 'aaf.pickle'
    modelfile_ctv = 'ctv.pickle'
    modelfile_pathseg = 'pathseg.pickle'
    modelfile_pathseg_post = 'pathseg_postmeno.pickle'
   
    
    model_coxph = p.load(open(modelfile_coxph, 'rb'))
    model_aam = p.load(open(modelfile_aam, 'rb'))
    model_ctv = p.load(open(modelfile_ctv, 'rb'))
    model_pathseg = p.load(open(modelfile_pathseg, 'rb')) 
    model_pathseg_post = p.load(open(modelfile_pathseg_post, 'rb'))
    
    app.run(debug=False, host='0.0.0.0')
