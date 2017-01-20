
import os
import numpy as np
from sklearn.externals import joblib
import cv2
from scipy.misc import imresize
import gist_feature
import hog_feature
import afine_search

def main_fun(img_arr):
	
	ori_res = search_logo(img_arr)	
	affine = afine_search.affine_transform(img_arr)
	affine_res = search_logo(affine)

	print ori_res
	print affine_res

	inter = set(ori_res).intersection(affine_res)
	print inter

	return inter


def search_logo(img_arr):
	
	img = imresize(img_arr, (47*2, 55*2), interp = 'bicubic')

	gist = gist_feature.feature(img)
	hog = hog_feature.hog_call(img)
	hog_list = predict_hog(hog)

	gist_list = predict_gist(gist)

	inter = set(gist_list).intersection(hog_list)
	#print list(inter)
	return list(inter)


def predict_hog(hog):
	profile_id = []
	hog_classifier = '../database/classifier/hog'
	hog_list = os.listdir(hog_classifier)
	for x, y in enumerate(hog_list):
		machine_path = os.path.join(hog_classifier, y)
		clf = joblib.load(machine_path)
		predict = clf.predict(hog)
		for k in predict:
			if k == 1:
				temp = ((machine_path.split('/')[4])).split('_')[0]
				profile_id.append(temp)
	return profile_id

def predict_gist(gist):
	profile_id = []
	gist_classifier = '../database/classifier/gist'
	gist_list = os.listdir(gist_classifier)
	for x, y in enumerate(gist_list):
		machine_path = os.path.join(gist_classifier, y)
		clf = joblib.load(machine_path)
		predict = clf.predict(gist)
		for k in predict:
			if k == 1:
				temp = ((machine_path.split('/')[4])).split('_')[0]
				profile_id.append(temp)
	return profile_id


if __name__ == '__main__':
	img_path = '/home/rahul/Downloads/logo_data/alfa romeo/Correct/images (13)_jpg_logo_500_56737r_1.png'
	img_arr = cv2.imread(img_path)
	main_fun(img_arr)