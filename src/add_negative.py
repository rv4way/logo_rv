

import numpy as np
import pandas as pd
import os
import add_logo
import random

'''
def negative(profile_id):

	neg_sel_gist = '../database/negative_selection/gist'
	neg_sel_hog = '../database/negative_selection/hog'
	
	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	pos_gist = os.listdir(positive_gist)
	pos_gist.remove(profile_id+'.csv')

	pos_hog = os.listdir(positive_hog)
	pos_hog.remove(profile_id+".csv")

	create_negative(pos_gist, positive_gist, negative_gist, profile_id, 'GIST')
	create_negative(pos_hog, positive_hog, negative_hog, profile_id, 'HOG')

	increase_postive(profile_id, positive_gist, negative_gist)
	increase_postive(profile_id, positive_hog, negative_hog)



def create_negative(profile_list, pos_location, neg_location, profile_id, method):

	neg_file = os.path.join(neg_location, profile_id+'.csv')

	for x, y in enumerate(profile_list):
		temp_data = pd.read_csv(os.path.join(pos_location, y), sep=',',header=None)
		temp_data = np.asarray(temp_data)
		temp = int(len(temp_data)/4)

		for k in range(temp):
			if method == str('HOG'):
				tem = (temp_data[k])[:360]
			elif method == str('GIST'):
				tem = (temp_data[k])[:512]
			tem = np.concatenate((tem,[0]))
			add_logo.write_csv(neg_file, tem)

def increase_postive(profile_id, pos_location, neg_location):

	neg_data = pd.read_csv(os.path.join(neg_location, profile_id+'.csv'), sep = ',', header = None)
	neg_data = np.asarray(neg_data)

	pos_data = pd.read_csv(os.path.join(pos_location, profile_id+'.csv'), sep = ',', header = None)
	pos_data = np.asarray(pos_data)

	itterate = len(neg_data) - len(pos_data)
	if itterate < 0:
		increase_negative(profile_id, pos_location, neg_location)
	for x in range(itterate):
		temp = random.choice(pos_data)
		add_logo.write_csv(os.path.join(pos_location, profile_id+'.csv'), temp)


def increase_negative(profile_id, pos_location, neg_location):

	neg_data = pd.read_csv(os.path.join(neg_location, profile_id+'.csv'), sep = ',', header = None)
	neg_data = np.asarray(neg_data)

	pos_data = pd.read_csv(os.path.join(pos_location, profile_id+'.csv'), sep = ',', header = None)
	pos_data = np.asarray(pos_data)

	itterate = len(pos_data) - len(neg_data)
	if itterate < 0:
		increase_postive(profile_id, pos_location, neg_location)
	for x in range(itterate):
		temp = random.choice(neg_data)
		add_logo.write_csv(os.path.join(neg_location, profile_id+'.csv'), temp)
'''

def main_fun(profile_id):

	neg_sel_gist = '../database/negative_selection/gist'
	neg_sel_hog = '../database/negative_selection/hog'
	
	negative_gist = '../database/negative/gist'
	negative_hog = '../database/negative/hog'

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	pos_gist = os.listdir(positive_gist)
	pos_gist.remove(profile_id+'.csv')
	create_negative_gist(profile_id, pos_gist, positive_gist, negative_gist)

	pos_hog = os.listdir(positive_hog)
	pos_hog.remove(profile_id+".csv")
	create_negative_hog(profile_id, pos_hog, positive_hog, negative_hog)

	check_equal(profile_id, positive_gist, negative_gist)

def create_negative_gist(profile_id, profile_list, pos_path, neg_path):

	neg_file = os.path.join(neg_path, str(profile_id+'.csv'))
	print neg_file

	for x, y in enumerate(profile_list):
		temp_data = pd.read_csv(os.path.join(pos_path, y), sep=',',header=None)
		temp_data = np.asarray(temp_data)
		temp = int(len(temp_data)/4)

		for k in range(temp):
			tem = (temp_data[k])[:512]
			tem = np.concatenate((tem,[0]))
			add_logo.write_csv(neg_file, tem)

def create_negative_hog(profile_id, profile_list, pos_path, neg_path):

	neg_file = os.path.join(neg_path, str(profile_id+'.csv'))

	for x, y in enumerate(profile_list):
		temp_data = pd.read_csv(os.path.join(pos_path, y), sep=',',header=None)
		temp_data = np.asarray(temp_data)
		temp = int(len(temp_data)/4)

		for k in range(temp):
			tem = (temp_data[k])[:360]
			tem = np.concatenate((tem,[0]))
			add_logo.write_csv(neg_file, tem)

def check_equal(profile_id, pos_path, neg_path):
	pos_file_path = os.path.join(pos_path, str(profile_id+'.csv'))
	neg_file_path = os.path.join(neg_path, str(profile_id+'.csv'))

	pos_data = pd.read_csv(pos_file_path, sep = ',', header = None)
	pos_data = np.asarray(pos_data)

	neg_data = pd.read_csv(neg_file_path, sep = ',', header = None)
	neg_data = np.asarray(neg_data)

	if len(neg_data) < len(pos_data):
		itter = len(pos_data) - len(neg_data)
		for x in range(itter):
			tem = random.choice(neg_data)
			add_logo.write_csv(neg_file_path, tem)

	elif len(neg_data) > len(pos_data):
		itter = len(neg_data) - len(pos_data)
		for x in range(itter):
			tem = random.choice(pos_data)
			add_logo.write_csv(pos_file_path, tem)