from flask import Flask,jsonify,Response,request, redirect, url_for
import json
from celery import Celery
from celery.utils.log import get_task_logger
from werkzeug.contrib.fixers import ProxyFix
import os
from werkzeug.utils import secure_filename
import cv2
import io
import numpy as np
import pandas as pd
from operator import is_not
from functools import partial
import add_logo
import search_logo

'''using celery tasks queue to queue the tsks and images'''



#logger = get_task_logger('/media/rahul/42d36b39-1ad7-45d4-86bb-bf4e0a66a97f/logo aws/DataBase/test.log')#get the log file instance
logger = get_task_logger(__name__)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['mp4', '3gp', 'avi'])


'''configuring celery and redis redis is our broker. '''
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0' #url where broker service is running
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
celery = Celery(app.name, backend=app.config['CELERY_RESULT_BACKEND'],broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@celery.task
def my_background_task(img_arr, profile_id): #this is the celery task this will execute the tasks in the background
	print 'HELL'
	try:
		logger.info('starting function %s'%(name))
		add_logo.add_logo(img_arr, profile_id)
		update_idcsv(profile_id)
		logger.info("completed %s"%(name))
	except Exception,e:
		print 'exceptiom',str(e)
	return "done"


#return the result
@app.route("/image-processing/search",methods=['GET', 'POST']) #this will check if the image is present or not
def hello():
	print 'HELOO*******'
	try:
		if request.method == 'POST':	
			photo = request.files['photo'] #if photo is present in the request object

			in_memory_file = io.BytesIO()
			photo.save(in_memory_file)
			data1 = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
			color_image_flag = 1

			img = cv2.imdecode(data1, color_image_flag) #img is the image
		
			v = search_logo.search_logo(img)
			print v
			if len(v) == 0:
				v = json.dumps(v)
				ret_val={'message':'images not found','status':0,'data':v}
				return 	jsonify(**ret_val)
			else:
				v = json.dumps(v)	
				ret_val={'message':'images found','status':1,'data':v}
				return 	jsonify(**ret_val)			
		
	except:
		ret_val={'message':'server error. database empty','status':0,'data':'No Data' }
		return 	jsonify(**ret_val)

#get status of added image
@app.route("/image-processing/logo/status",methods=['GET', 'POST'])
def get_status():
	try:
		if request.method == 'POST':
			
			header_req = request.headers.get('x-image-profile-id')
			c = check_status(str(header_req))
			if c == True:
				ret_val={'message':'image is added.','status':1,'data':header_req }
				return 	jsonify(**ret_val)
			else:
				ret_val={'message':'image not added.','status':0,'data':header_req }
				return 	jsonify(**ret_val)		
	except:
		ret_val={'message':'can not be added','status':0,'data':header_req }
		return 	jsonify(**ret_val)

#add the image to thedata base
@app.route("/image-processing/logo/add",methods=['GET','POST']) #address at which to send if image is not present in data baase
def add():
	try:
		if request.method == 'POST':
			header_req = request.headers.get('x-image-profile-id')
			photo = request.files['photo']
			in_memory_file = io.BytesIO()
			photo.save(in_memory_file)
			data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
			color_image_flag = 1
			img = cv2.imdecode(data, color_image_flag)
			profile_id = str(header_req)
					
			my_background_task.delay(img, profile_id)

			ret_val={'message':'image queued for adding.','status':2,'data':header_req }
			return 	jsonify(**ret_val)
	except:
		ret_val={'message':'request cannot be processed','status':0,'data':header_req }
		return 	jsonify(**ret_val)

def update_idcsv(name):
	search_in = pd.read_csv(data["ImageCsv"], sep = ',', header = None)
	search_in = np.asarray(search_in)
	if name in search_in[0]:
		return
		#update
	else:
		temp_csv_path = data["ImageCsv"]
		f = open(temp_csv_path, 'a')
		f.write(name)
		f.write('\n')
		f.close()

def check_status(name):
	search_in = pd.read_csv(data["ImageCsv"],sep=',',header=None)
	search_in2 = np.asarray(search_in)
	if name in search_in2:
		return True
	else:
		return False

app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == "__main__":
	app.run(debug = True, host = '0.0.0.0')