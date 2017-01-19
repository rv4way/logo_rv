

import numpy as np
import pandas as pd
import os
import add_logo
import random


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