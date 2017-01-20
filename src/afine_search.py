import numpy as np
import cv2
import salt_pepper
def affine_transform(img):
	#print profile_path, name
	rows,cols,ch = img.shape

	x_rot = 50
	y_rot = 200

	pts1 = np.float32([[50,50],[200,50],[50,200]])

	x_rot = x_rot + 20
	y_rot = y_rot + 20

	pts2 = np.float32([[x_rot, x_rot],[200,50],[x_rot,y_rot]])

	M = cv2.getAffineTransform(pts1,pts2)

	dst = cv2.warpAffine(img,M,(cols,rows))
	salt = salt_pepper.noise_addition(dst)
	con_img = convolve_image(salt)
	return con_img

def convolve_image(img):

	kernel = np.array([ [0,-1,0],
                    [-1,4.5,-1],
                    [0,-1,0] ],np.float32)

	new_img = cv2.filter2D(img,-1,kernel)
	return new_img

def inside_logo(img_arr):

	x, y, z = img_arr.shape
	x1 = int(x*0.2)
	y1 = int(y*0.2)
	in_logo = img_arr[x1:x-x1, y1:y-y1]
	return in_logo