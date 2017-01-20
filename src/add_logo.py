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
import histogram
import time


def add_logo(img_arr, profile_id):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	img = imresize(img_arr, (47*2, 55*2), interp = 'bicubic')

	histogram.add_hist(img, profile_id)

	gist = gist_feature.feature(img)
	gist = np.concatenate((gist,[1]))
	gist_file = positive_gist + '/' + str(profile_id) + '.csv'
	write_csv(gist_file, gist)
	print 'GIST CALCULATED'

	hog = hog_feature.hog_call(img)	
	hog = np.concatenate((hog,[1]))	
	hog_file = positive_hog + '/' + str(profile_id) + '.csv'	
	write_csv(hog_file, hog)
	print 'HOG CALCULATED'

	print 'GENERATING AFFINE'
	no_affine = affine_transform.generate_affine(img, profile_id)
	print 'AFFINE GENERATED'
	
	add_negative.main_fun(profile_id)
	print 'NEGATIVE ADDED'
	
	train.main_fun(profile_id)
	print 'MACHINE TRAINED'

	re_train.main_fun(profile_id)
	print 'ALL MACHINE RE_TRAINED'


def write_csv(file_path, csv_data):

	with open(file_path, 'a') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(csv_data)



if __name__ == '__main__':
	x = time.ctime()
	img_path = '/home/rahul/Downloads/7up/Correct/download (1)_jpg_logo_400_5603r_1.png'
	img_arr = cv2.imread(img_path)
	add_logo(img_arr, '7upLogo')
	y = time.ctime()

	print x
	print y