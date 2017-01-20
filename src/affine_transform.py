

from __future__ import division
import salt_pepper as sp
import cv2
import numpy as np
import random
import math
import os
import csv
import pandas as pd
from scipy.misc import imresize
import add_logo
import gist_feature
import hog_feature


def generate_affine(img, profile_id):

	positive_gist = '../database/positive/gist'
	positive_hog = '../database/positive/hog'

	for i in range(50):        
		#padd the image
		img1 = padding(img)
		#cv2.imshow('after concatat',img1)
		gamma_1 = random_1()
		gamma_main = random.randrange(1,3,1)
		gamma_last = gamma_main + gamma_1
		noise_addition = adjust_gamma(img1, gamma=gamma_last)#change the gammmaa of the image randomly
		noise_ad2 = sp.noise_addition(noise_addition) #add the salt and peppr noise to the image
		pts1 = np.float32([[50,50],[200,50],[50,200]])
		(h, w) = img1.shape[:2]
		rows2,cols2,ch2 = img1.shape #ch=3 rows = hieght and col is width

		#creating the rotation matrix
		x=random.randrange(-5,5,1)

		y= random.random()
		z=x+y
		x=math.radians(z)

		sin=math.sin(x)
		cos=math.cos(x)
		rotation = [[cos,sin],[-sin,cos]] #2x2 rotation matrix
		#shear matrices

		shear_fact_x = -0.3+random_1() #take random values for both x and y
		shear_fact_y = -0.3+random_1()
		shear_x = [[1,shear_fact_x],[0,1]]
		shear_y = [[1,0],[shear_fact_y,1]]
		shear_new = np.array(shear_x,dtype=np.float32)
		rotation_new = np.array(rotation,dtype=np.float32)


		#multiplying 2D matrices
		afine_t= np.dot(shear_new, rotation_new)
		shear_new = np.array(shear_y,dtype=np.float32)
		afine_t2 = np.dot(afine_t,shear_new)
		#afine_t = shear_new
		pts2 = np.dot(pts1,afine_t2)#new points use for warping the image
		#print pts2
		#cv2.imshow('input',img)
		s11,s12,s13 = noise_ad2.shape
		for r in range(s11):
			for c in range(s12):
				if np.all(noise_ad2[r,c,:]==0):
					noise_ad2[r,c,0]=noise_ad2[r,c,0]+1
					noise_ad2[r,c,1]=noise_ad2[r,c,1]+1
					noise_ad2[r,c,2]=noise_ad2[r,c,2]+1        
		M = cv2.getAffineTransform(pts1,pts2)
		dst = cv2.warpAffine(noise_ad2,M,(cols2,rows2))#get afine of image
		#cv2.imshow('imshow',dst)
		s1,s2,s3 = dst.shape
		#removing the extra black areas from the images so that only logos is visible.

		count = 0
		count2 = 0
		for r in range(s1):
			num = count_rows(dst[r,:,:])
			if num >= 30:
				count=count+1
				#break
			else:
				#count = count+1
				break
		for c in range(s2):
			num2 = count_rows(dst[:,c,:])
			if(num2>=30):
				count2 =count2+1
				#break
			else:
				#count2 = count2+1
				break
		r1  = s1-1
		while(r1>=0):
			num3 = count_rows(dst[r1,:,:])
			if(num3 >= 30):
				#break
				r1=r1-1
			else:
				#r1=r1-1
				break
		c1 = s2-1
		while(c1>=0):
			num4 = count_rows(dst[:,c1,:])
			if (num4 >=30):
				#break
				c1=c1-1
			else:
				#c1 = c1-1 
				break    
		dst2 = dst[count:r1,count2:c1]
		count=r1=c1=count2=0
		s1,s2,s3 = dst2.shape

		#add the gist and hog feature of the image
		if s2 !=0 and s1 !=0:
			
			gist = gist_feature.feature(dst2)
			gist = np.concatenate((gist,[1]))
			gist_file = positive_gist + '/' + str(profile_id) + '.csv'
			add_logo.write_csv(gist_file, gist)

			hog = hog_feature.hog_call(dst2)	
			hog = np.concatenate((hog,[1]))	
			hog_file = positive_hog + '/' + str(profile_id) + '.csv'	
			add_logo.write_csv(hog_file, hog)

	return i

def padding(img):
	x,y,z = img.shape
	#--take 30% of rows--#    
	temp_x = round(0.4*x)
	temp_y = round(0.4*y)
	#--take the regions of start and end which needs to be padd along x axis--#
	x_region_end = img[:,(y-2):,:]
	x_region_start = img[:,:2,:]
	#--padd the region in x axis--#
	for i in range(50):        
		img = np.concatenate((img, x_region_start), axis=1)
		img = np.concatenate((x_region_end,img),axis=1)   
	#--padding for y axis--#
	y_region_end = img[(x-1):,:,:]
	y_region_start = img[:1,:,:]
	for i in  range(50):
		img = np.concatenate((y_region_start,img),axis=0)
		img = np.concatenate((img,y_region_end),axis=0)
	return img

def random_1():    
	z = random.random()*0.6       
	return z

def adjust_gamma(image, gamma=1.0):	
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def count_rows(arr):
	rows,col = arr.shape
	num_zeros=0
	for i in range(rows):
		if np.all(arr[i,:]==0):
			num_zeros = num_zeros+1
	ret = (num_zeros/rows)*100
	return ret