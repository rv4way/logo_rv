

import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import random
import add_negative

'''
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
'''


def main_fun(profile_id):

	clasifier_gist = '../database/classifier/gist'
	clasifier_hog = '../database/classifier/hog'

	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	pos_gist_file = os.path.join(positive_gist, str(profile_id+'.csv'))
	neg_gist_file = os.path.join(negative_gist, str(profile_id+'.csv'))

	pos_hog_path = os.path.join(positive_hog, str(profile_id+'.csv'))
	neg_hog_path = os.path.join(negative_hog, str(profile_id+'.csv'))

	pos_gist_data = pd.read_csv(pos_gist_file, sep = ',', header = None)
	pos_gist_data = np.asarray(pos_gist_data)

	neg_gist_data = pd.read_csv(neg_gist_file, sep = ',', header = None)
	neg_gist_data = np.asarray(neg_gist_data)

	pos_hog_data = pd.read_csv(pos_hog_path, sep = ',', header = None)
	pos_hog_data = np.asarray(pos_hog_data)

	neg_hog_data = pd.read_csv(neg_hog_path, sep = ',', header = None)
	neg_hog_data = np.asarray(neg_hog_data)

	gist_classifier_path = os.path.join(clasifier_gist, str(profile_id+'_0.pkl'))
	hog_classifier_path = os.path.join(clasifier_hog, str(profile_id+'_0.pkl'))

	create_classifier(profile_id, pos_gist_data, neg_gist_data, gist_classifier_path)
	create_classifier(profile_id, pos_hog_data, neg_hog_data, hog_classifier_path)

	new_neg_gist = predict_negative(profile_id, positive_gist, clasifier_gist)
	new_neg_hog = predict_negative(profile_id, positive_hog, clasifier_hog)
	
	re_train_classifier(profile_id, new_neg_gist, positive_gist, negative_gist, clasifier_gist)
	re_train_classifier(profile_id, new_neg_hog, positive_hog, negative_hog, clasifier_hog)

def create_classifier(profile_id, pos_data, neg_data, classifier_path):	
	train_data = []
	train_label =[]
	for x in range(len(pos_data)):
		train_data.append(pos_data[x][:-1])
		train_label.append(int(pos_data[x][-1]))

	for x in range(len(neg_data)):
		train_data.append(neg_data[x][:-1])
		train_label.append(int(neg_data[x][-1]))

	imp = Imputer(strategy = 'median')
	imp.fit(train_data)
	train_data = imp.transform(train_data)

	est = RandomForestClassifier(n_estimators =20,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,n_jobs=1)

	est.fit(train_data, train_label)
	joblib.dump(est, classifier_path, compress=9)

def predict_negative(profile_id, positive_path, classifier_path):
	
	pos_list = os.listdir(positive_path)
	pos_list.remove(profile_id+'.csv')

	machine_path = os.path.join(classifier_path, str(profile_id+'_0.pkl'))
	clf = joblib.load(machine_path)

	re_train_data = []
	for x, y in enumerate(pos_list):
		file_path = os.path.join(positive_path, y)
		neg_data = pd.read_csv(file_path, sep = ',', header = None)
		neg_data = np.asarray(neg_data)
		tem = []
		for k in neg_data:
			tem.append(k[:-1])
		predict = clf.predict(tem)
		for k, l in enumerate(predict):
			if l == 1:
				new_tem = tem[k]
				new_tem = np.concatenate((new_tem,[0]))
				re_train_data.append(new_tem)
	return re_train_data

def re_train_classifier(profile_id, new_neg_data, pos_path, neg_path, classifier_path):

	neg_file_path = os.path.join(neg_path, str(profile_id+'.csv'))
	pos_file_path = os.path.join(pos_path, str(profile_id+'.csv'))

	pos_data = pd.read_csv(pos_file_path, sep = ',', header = None)
	pos_data = np.asarray(pos_data)

	neg_data = pd.read_csv(neg_file_path, sep = ',', header = None)
	neg_data = np.asarray(neg_data)

	new_neg = []
	new_pos =[]
	for x in pos_data:
		new_pos.append(x)
	for x in range(len(new_neg_data)):
		new_pos.append(random.choice(pos_data))
	
	for x in neg_data:
		new_neg.append(x)
	for x in new_neg_data:
		new_neg.append(x)

	new_classifier_path = os.path.join(classifier_path, str(profile_id+'_1.pkl'))
	create_classifier(profile_id, new_pos, new_neg, new_classifier_path)