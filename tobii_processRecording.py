"""
Process as Tobii Glasses 2 recording
-jjm35

Tested with Python 3.6, open CV 3.2. Might work with others

Since the data originates on a SD card (or temp directory somewhere), a new output directory
will be created for each recording, with the path specified by the user. 

The results of the processing will be stored withing a "processed" directory within this new output directory
"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

import sys, os, shutil
import matplotlib
matplotlib.use('tkagg')
from os.path import join
import cv2
import numpy as np 
import pandas as pd 
import argparse
import time

from tobii_gazeDataFormatting import copyTobiiRecording, formatGazeData

OPENCV3 = (cv2.__version__.split('.')[0] == '3')
print("OPENCV version " + cv2.__version__)


"""
This script can be called to process recordings by Tobii Glasses 2

Usage:
	python tobii_processRecording.py <path to input dir> <path to output directory> <path to reference stimulus> <path to camera calibration JSON>

Inputs:
	- recording directory: path to recording session directory (either raw data on SD card, or a local copy of that data on harddrive)
	- output directory: path to where you want the data copied to and where the 'processed' results written
	- reference stimulus: path to a decently high resolution jpg of the stimulus you are attempting to map gaze on to
	- camera calibration: path to the camera calibration JSON file for this make/model of eye-tracker

Outputs:
	- all outputs written to "processed" directory created within the specified output directory
	- movies (all output movies for the specified frames only):
		- orig
		- orig + gaze overlay
		- orig + evolving heatmap overlay
		- ref + evolving heatmap overlay
	- data:
		- camGazePosition: camera locations
		- camGazePosition_smooth: spike removal and smoothing

Note about coordinates:
The operations below focus on translating between multiple different coordinate systems.
For clarity, here are those different systems and the labels used when referencing each:
	- World Frame (world):		Coordinate system used on frames take from the camera (e.g. world camera on glasses)
								This is a 2D system with the origin in the bottom-left corner of the frame, units in pixels
	- Reference Image (ref):	Coordinate system of the reference image that is expected to be found in the camera frame. 
								This is a 2D system with the origin in the top-left corner, units in pixels
	- Object (obj):				Coordinate system of the reference image AS IT OCCURS IN THE PHYSICAL WORLD. This is a 3D
								system, however all z-values are set to 0, defining the plane of the wall that the image is 
								on. The units are in whatever physical coordinates you desire (e.g. inches); whatever unit you 
								set will be used on subsequent measurements, such as camera position. 
"""


def processRecording(inputDir, refFile, cameraCalib):
	"""
	Open the recording from the input dir
	Format the pupil data, 
	Loop through each frame of the recording and create output videos
	"""
	# Settings:
	framesToUse = np.arange(0, 445, 1)

	# start time
	process_startTime = time.time()

	# specify an output dir for the 'processed' results
	outputDir = join(inputDir, 'processed')
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	### Prep the gaze data ################################
	print('Prepping gaze data...')
	# format pupil data
	formatGazeData(inputDir)




###
if __name__ == '__main__':
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('inputDir', help='path to the raw recording dir (e.g. SD card)')
	parser.add_argument('outputDir', help='path to where output data copied and saved to')
	parser.add_argument('referenceFile', help='path to reference stimuli')
	parser.add_argument('cameraCalibration', help='path to camera calibration file')
	args = parser.parse_args()

	# Check if input directory is valid
	if not os.path.isdir(args.inputDir):
		print('Invalid input dir: ' + args.inputDir)
		sys.exit()
	else:
		# copy the raw data to the output dir
		newDataDir = copyTobiiRecording(args.inputDir, args.outputDir)

		# processing the data (note: the output dir from previous step is now the input dir for this step)
		processRecording(newDataDir, args.referenceFile, args.cameraCalibration)




