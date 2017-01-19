

import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import random
import add_negative


def make_log(profile_id, log_detail):
	log_path = '../database/classifier_log.csv'
	with open(log_path, 'a') as log:
		line = str(profile_id) + ',' + str(log_detail) + '\n'
		log.write(line)
		log.write('\n')

def merge_data(profile_id):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	gist_classifier = '../database/classifier/gist'
	hog_classifier = '../database/classifier/hog'

	pos_gist_path = os.path.join(positive_gist, profile_id+'.csv')
	pos_hog_path = os.path.join(positive_hog, profile_id+'.csv')

	neg_gist_path = os.path.join(negative_gist, profile_id+'.csv')
	neg_hog_path = os.path.join(negative_hog, profile_id+'.csv')

	pos_gist_data = pd.read_csv(pos_gist_path, sep = ',', header = None)
	pos_gist_data = np.asarray(pos_gist_data)

	neg_gist_data = pd.read_csv(neg_gist_path, sep = ',', header = None)
	neg_gist_data = np.asarray(neg_gist_data)

	pos_hog_data = pd.read_csv(pos_hog_path, sep = ',', header = None)
	pos_hog_data = np.asarray(pos_hog_data)

	neg_hog_data = pd.read_csv(neg_hog_path, sep = ',', header = None)
	neg_hog_data = np.asarray(neg_hog_data)

	gist_classifier_path = os.path.join(gist_classifier, str(profile_id+'_0.pkl'))
	hog_classifier_path = os.path.join(hog_classifier, str(profile_id+'_0.pkl'))

	create_gist_classifier(profile_id, pos_gist_data, neg_gist_data, gist_classifier_path)
	create_hog_classifier(profile_id, pos_hog_data, neg_hog_data, hog_classifier_path)



def create_gist_classifier(profile_id, pos_data, neg_data, path):

	train_data = []
	train_label = []
	if len(pos_data) != len(neg_data):
		make_log(profile_id, 'NOT_ADDED, NEG-POS_MISSMATCH_FOR_GIST')
	else:
		for x in range(len(pos_data)):
			train_data.append(pos_data[x][:512])
			train_label.append(int(pos_data[x][512]))
		for x in range(len(neg_data)):
			train_data.append(neg_data[x][:512])
			train_label.append(int(neg_data[x][512]))

	est = RandomForestClassifier(n_estimators =20,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,n_jobs=1)

	est.fit(train_data, train_label)

	joblib.dump(est, path, compress=9)

	make_log(profile_id, 'PROFILE_ADDED_FOR_GIST')


def create_hog_classifier(profile_id, pos_data, neg_data, path):

	train_data = []
	train_label = []
	if len(pos_data) != len(neg_data):
		make_log(profile_id, 'NOT_ADDED, NEG-POS_MISSMATCH_FOR_HOG')
	else:
		for x in range(len(pos_data)):
			train_data.append(pos_data[x][:360])
			train_label.append(int(pos_data[x][360]))
		for x in range(len(neg_data)):
			train_data.append(neg_data[x][:360])
			train_label.append(int(neg_data[x][360]))

	est = RandomForestClassifier(n_estimators =20,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,n_jobs=1)

	est.fit(train_data, train_label)
	joblib.dump(est, path, compress=9)
	make_log(profile_id, 'PROFILE_ADDED_FOR_HOG')

