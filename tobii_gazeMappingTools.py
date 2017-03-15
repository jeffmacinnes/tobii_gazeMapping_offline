""" 
Set of tools for mapping between world coordinates and camera coordinates on still images or video frames 

Note about coordinates:
The operations below focus on translating between multiple different coordinate systems.
For clarity, here are those different systems and the labels used when referencing each:
	- Camera Frame (frame):		Coordinate system used on frames take from the camera (e.g. scene camera on glasses)
								This is a 2D system with the origin in the top-left corner of the frame, units in pixels
	- Reference Image (ref):	Coordinate system of the reference image that is expected to be found in the camera frame. 
								This is a 2D system with the origin in the top-left corner, units in pixels
	- Object (obj):				Coordinate system of the reference image AS IT OCCURS IN THE PHYSICAL WORLD. This is a 3D
								system, however all z-values are set to 0, defining the plane of the wall that the image is 
								on. The units are in whatever physical coordinates you desire (e.g. inches); whatever unit you 
								set will be used on subsequent measurements, such as camera position. 

-jjm 7/2016
"""
from __future__ import division
from __future__ import print_function
import os
import sys
import cv2
import json
import numpy as np 

OPENCV3 = (cv2.__version__.split('.')[0] == '3')

class GazeMapper:
	def __init__(self, camera_calibration, ref_image, obj_image_size):
		"""
		camera_calibration: camera calibration json file
		ref_image: reference image file (i.e. image to search for in each camera image or video frame) 
		ref_image_size: (height, width) tuple specifying the ref image dimensions in world units (note: these units will set
			the units for all calculations later on. World origin will be top left corner of the image)
		"""
		# Load the calibration file, extract camera intrinsic matrix and distorton coefs
		self.calibFile = camera_calibration
		with open(self.calibFile, 'r') as infile:
			dictValues = json.load(infile)
		self.camMtx = np.asarray(dictValues['camera_matrix'])
		self.distCoefs = np.asarray(dictValues['dist_coefs'])

		# Load the ref image, calculate pixels/unit
		self.refImg = ref_image
		self.pixPerUnit = self.refImg.shape[0]/obj_image_size[0]		# ref image mapping b/w pixels and world units

		# Create a copy of the original refImg in color; convert refImg to grayscale
		self.refImgColor = self.refImg.copy()
		self.refImg = cv2.cvtColor(self.refImg, cv2.COLOR_BGR2GRAY)

		# Initialize the feature detection object, and matching object (choose which algorithm you want)
		if OPENCV3:
			self.featureDetect = cv2.xfeatures2d.SIFT_create()
		else:
			self.featureDetect = cv2.SIFT()
		self.min_match_count = 50
		self.min_good_matches = 4
		self.num_matches = 2 
		self.FLANN_INDEX_KDTREE = 0
		self.distance_ratio = 0.5 					# 0-1; lower values more conservative
		self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
		self.search_params = dict(checks=10)		# lower = faster, less accurate
		self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)

		# Find features on the reference image
		self.refImg_kp, self.refImg_des = self.findFeatures(self.refImg)
		print('Reference Image: found %s keypoints' % len(self.refImg_kp))

	def undistortFrame(self, frame):
		"""
		Undistort the supplied frame according to parameters in the camera calibration
		"""
		undistortedFrame = cv2.undistort(frame, self.camMtx, self.distCoefs)
		return undistortedFrame

	def findFeatures(self, img):
		"""
		Given a grayscale input image, find the locations of the features using whichever algorithm defined in __init__ 
		"""
		kp, des = self.featureDetect.detectAndCompute(img, None)
		return kp, des


	def findMatches(self, frame, frame_kp, frame_des):
		"""
		Find the matches between the descriptors for reference image (already calculated) and frame. 
			Inputs: 	camera frame key points, camera frame descriptors
			Output: 	2D coords of quaifying matches on reference image, 2D coords of qualifying matches on camera frame
		"""
		
		# find all matches
		matches = self.matcher.knnMatch(self.refImg_des, frame_des, k=self.num_matches)

		# filter out cases where the 2 matches (best guesses) are too close to each other
		goodMatches = []
		for m,n in matches:
			if m.distance < self.distance_ratio*n.distance:
				goodMatches.append(m)
		if len(goodMatches) > self.min_good_matches:
			refImg_pts = np.float32([self.refImg_kp[i.queryIdx].pt for i in goodMatches])
			frameImg_pts = np.float32([frame_kp[i.trainIdx].pt for i in goodMatches])

			return refImg_pts, frameImg_pts
		
		else:
			return None, None

	def ref2obj(self, refPts):
		"""
		convert the supplied refPts to their corresponding objPt representation
		"""
		# convert the refPts to objPts (i.e. from pixels to world coordinates)
		objPts = refPts/self.pixPerUnit

		# add Z-dimension to the obj coords (Z=0; Z defined as the plane of the wall the object is attached to)
		objPts = np.append(objPts, np.zeros(shape=(objPts.shape[0],1)), axis=1)
		return objPts
	

	def PnP_3Dmapping(self, refPts, framePts):
		"""
		solve for 3D mapping between camera and refence image in the world 
		Inputs:		2D pts for reference image, corresponding 2D pts on camera frame
		Output: 	rotation and translation vectors for describing the object relative to the camera
		"""
		
		# convert the refPts to objPts (i.e. from pixels to world coordinates)
		objPts_3D = self.ref2obj(refPts)

		# solve perspective-n-point problem
		ret, rvec, tvec, inliers = cv2.solvePnPRansac(objPts_3D, framePts, self.camMtx, self.distCoefs)
		return rvec, tvec


	def getCameraPosition(self, rvec, tvec):
		"""
		Given the rvec and tvec that will map object relative to camera, figure out camera relative to object
		InputsL rotation vector from solvePnP, translation vector from solvePnP
		Outputs: 	camPositon - [X, Y, Z] in world coord sytem
					camOrientation - [theta, x, y, z] where theta is rotation radians, and [x,y,z] are rotation axis
		"""	

		# Calculate the camera location given the rotation and translation vectors
		rotM = cv2.Rodrigues(rvec)[0]
		camPosition = -np.matrix(rotM).T * np.matrix(tvec)
		camPosition = np.squeeze(np.asarray(camPosition))							# flatten to 1D 

		# Calculate the camera orientation
		rotVector = np.squeeze(rvec*-1)												# invert and flatten to 1D array
		theta = np.sqrt(rotVector[0]**2 + rotVector[1]**2 + rotVector[2]**2)		# calculate angle of rotation
		rotAxis = rotVector/theta													# calculate the axis of rotration
		camOrientation = np.array([theta, rotAxis[0], rotAxis[1], rotAxis[2]])
		
		# return camera positon and orientation 
		return camPosition, camOrientation


	def get2Dmapping(self, srcPts, dstPts):
		"""
		Use findHomography to get a transform from the 2D source points to the corresponding 2D dest points.
		Will return a src2dst tranformation matrix, as well as the inverted dst2src matrix
		"""
		src2dst_transform, mask = cv2.findHomography(srcPts.reshape(-1,1,2), dstPts.reshape(-1,1,2), cv2.RANSAC, 5.0)
		dst2src_transform = cv2.invert(src2dst_transform) 	# note: the transformation matrix is in index=1 of the returned var
		return src2dst_transform, dst2src_transform[1]


	def projectImage2D(self, origFrame, transform2D, newImage):
		"""
		Will warp the new Imag according to the supplied transformation matrix and write into the original frame
		"""
		# warp the new image to the video frame
		warpedImage = cv2.warpPerspective(newImage, transform2D, origFrame.T.shape[1:])

		# mask and subtract new image from video frame
		warpedImage_bw = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
		if warpedImage.shape[2] == 4:
			alpha = warpedImage[:,:,3]
			alpha[alpha == 255] = 1 			# create mask of non-transparent pixels
			warpedImage_bw =cv2.multiply(warpedImage_bw, alpha)

		ret, mask = cv2.threshold(warpedImage_bw, 10, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		origFrame_bg = cv2.bitwise_and(origFrame, origFrame, mask=mask_inv)

		# mask the warped new image, and add to the masked background frame
		warpedImage_fg = cv2.bitwise_and(warpedImage[:,:,:3], warpedImage[:,:,:3], mask=mask)
		newFrame = cv2.add(origFrame_bg, warpedImage_fg)

		# return the warped new frame
		return newFrame

		
	def mapCoords2D(self, coords, transform2D):
		"""
		Will map the supplied coords to a new coordinate system using the supplied transformation matrix
		"""
		coords = np.array(coords).reshape(-1,1,2)
		mappedCoords = cv2.perspectiveTransform(coords, transform2D)
		mappedCoords = np.round(mappedCoords.ravel())

		return mappedCoords[0], mappedCoords[1] 



		


		
