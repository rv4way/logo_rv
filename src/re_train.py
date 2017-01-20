from __future__ import division
#from sklearn.cross_validation import train_test_split
import numpy as np
#from sklearn.preprocessing import Imputer
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from sklearn.externals import joblib
import random
import train
import train_basic

'''
def predict_neg(profile_id, method):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	pos_gist = os.listdir(positive_gist)
	pos_gist.remove(profile_id+'.csv')

	pos_hog = os.listdir(positive_hog)
	pos_hog.remove(profile_id+'.csv')

	neg_gist_file = os.path.join(negative_gist, str(profile_id+'.csv'))
	neg_hog_file = os.path.join(negative_hog, str(profile_id+'.csv'))

	if method == str('GIST'):
		classifier_path = os.path.join('../database/classifier/gist', str(profile_id+'_0.pkl'))
		clf = joblib.load(classifier_path)
		for x, y in enumerate(pos_gist):
			pos_gist_data = pd.read_csv(os.path.join(positive_gist, y), sep = ',', header = None)
			pos_gist_data = np.asarray(pos_gist_data)
			for z in pos_gist_data:
				tem = z[:512]
				predict = clf.predict(tem)
				for k in predict:
					if k == 1:
						tem = np.concatenate((tem,[0]))
						add_logo.write_csv(neg_gist_file, tem)


	if method == str('HOG'):
		classifier_path = os.path.join('../database/classifier/hog', str(profile_id+'_0.pkl'))
		clf = joblib.load(classifier_path)
		for x, y in enumerate(pos_hog):
			pos_hog_data = pd.read_csv(os.path.join(positive_hog, y), sep = ',', header = None)
			pos_hog_data = np.asarray(pos_hog_data)
			for z in pos_hog_data:
				tem = z[:360]
				predict = clf.predict(tem)
				for k in predict:
					if k == 1:
						tem = np.concatenate((tem,[0]))
						add_logo.write_csv(neg_hog_file, tem)

def add_positive(profile_id):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	pos_gist_path = os.path.join(positive_gist, str(profile_id+'.csv'))
	pos_hog_path = os.path.join(positive_hog, str(profile_id+'.csv'))

	neg_gist_path = os.path.join(negative_gist, str(profile_id+'.csv'))
	neg_hog_path = os.path.join(negative_hog, str(profile_id+'.csv'))

	pos_gist_data = pd.read_csv(pos_gist_path, sep = ',', header = None)
	pos_gist_data = np.asarray(pos_gist_data)

	neg_gist_data = pd.read_csv(neg_gist_path, sep = ',', header = None)
	neg_gist_data = np.asarray(neg_gist_data)

	pos_hog_data = pd.read_csv(pos_hog_path, sep = ',', header = None)
	pos_hog_data = np.asarray(pos_hog_data)

	neg_hog_data = pd.read_csv(neg_hog_path, sep = ',', header = None)
	neg_hog_data = np.asarray(neg_hog_data)

	if len(neg_gist_data) > len(pos_gist_data):		
		itterate = len(neg_gist_data) - len(pos_gist_data)
		for x in range(itterate):
			temp = random.choice(pos_gist_data)
			add_logo.write_csv(os.path.join(positive_gist, profile_id+'.csv'), temp)

	elif len(neg_gist_data) < len(pos_gist_data):
		itterate = len(pos_gist_data) - len(neg_gist_data)
		for x in range(itterate):
			temp = random.choice(neg_gist_data)
			add_logo.write_csv(os.path.join(negative_gist, profile_id+'.csv'), temp)


	if len(neg_hog_data) > len(pos_hog_data):		
		itterate = len(neg_hog_data) - len(pos_hog_data)
		for x in range(itterate):
			temp = random.choice(pos_hog_data)
			add_logo.write_csv(os.path.join(positive_hog, profile_id+'.csv'), temp)

	elif len(neg_hog_data) < len(pos_hog_data):
		itterate = len(pos_hog_data) - len(neg_hog_data)
		for x in range(itterate):
			temp = random.choice(neg_hog_data)
			add_logo.write_csv(os.path.join(negative_hog, profile_id+'.csv'), temp)


def re_train_classifier(profile_id):
	positive_gist = '../database/positive/gist'
	negative_gist = '../database/negative/gist'

	positive_hog = '../database/positive/hog'
	negative_gist = '../database/negative/hog'

	profile_list_gist = os.listdir(positive_gist)
	profile_list_gist.remove(profile_id+'.csv')
	for x in profile_list_gist:
		x = x.split('.')[0]
		start(x)

def start(profile_id):

	add_negative.negative(profile_id)
	train.merge_data(profile_id)
	predict_neg(profile_id, 'GIST')
	predict_neg(profile_id, 'HOG')
	add_positive(profile_id)
	train.merge_data(profile_id)
'''
def main_fun(profile_id):

 	clasifier_gist = '../database/classifier/gist'
	clasifier_hog = '../database/classifier/hog'

	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	pos_gist_list = os.listdir(positive_gist)
	pos_hog_list = os.listdir(positive_hog)

	for x in pos_gist_list:
		profile_id = x.split('.')[0]
		#print profile_id
		new_neg = train.predict_negative(profile_id, positive_gist, clasifier_gist)
		train.re_train_classifier(profile_id, new_neg, positive_gist, negative_gist, clasifier_gist)

	for x in pos_hog_list:
		profile_id = x.split('.')[0]
		new_neg = train.predict_negative(profile_id, positive_hog, clasifier_hog)
		train.re_train_classifier(profile_id, new_neg, positive_hog, negative_hog, clasifier_hog)
		train_basic.start_1(profile_id)
		