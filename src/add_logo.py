import cv2
import os
import numpy as np
import cPickle
import gist_feature
import hog_feature
import csv
from scipy.misc import imresize
import affine_transform
import add_negative
import train
import re_train


def add_logo(img_arr, profile_id):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	img = imresize(img_arr, (47*2, 55*2), interp = 'bicubic')

	gist = gist_feature.feature(img)
	gist = np.concatenate((gist,[1]))
	gist_file = positive_gist + '/' + str(profile_id) + '.csv'
	write_csv(gist_file, gist)

	hog = hog_feature.hog_call(img)	
	hog = np.concatenate((hog,[1]))	
	hog_file = positive_hog + '/' + str(profile_id) + '.csv'	
	write_csv(hog_file, hog)

	no_affine = affine_transform.generate_affine(img, profile_id)

	add_negative.negative(profile_id)

	train.merge_data(profile_id)

	re_train.predict_neg(profile_id, 'GIST')
	re_train.predict_neg(profile_id, 'HOG')

	re_train.add_positive(profile_id)
	train.merge_data(profile_id)

	re_train.re_train_classifier(profile_id)

	return 'Done'


def write_csv(file_path, csv_data):

	with open(file_path, 'a') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(csv_data)



if __name__ == '__main__':
	img_path = '/home/rahul/Desktop/images/logos/hp.png'
	img_arr = cv2.imread(img_path)
	add_logo(img_arr, 'testLogo')