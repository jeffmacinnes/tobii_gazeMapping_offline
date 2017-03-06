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
import fileinput
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from os.path import join
import cv2
import numpy as np 
import pandas as pd 
import argparse
import time
from scipy.ndimage import filters

from tobii_gazeDataFormatting import copyTobiiRecording, formatGazeData

# custom tobii projection tools
import tobii_gazeMappingTools as gm

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

# global settings/vars
obj_dims = (30,20) 		# real world dims (height, width) in inches of the stimulus

def processRecording(inputDir, refFile, cameraCalib):
	"""
	Open the recording from the input dir
	Format the pupil data, 
	Loop through each frame of the recording and create output videos
	"""
	# Settings:
	framesToUse = np.arange(0, 350, 1)
	#framesToUse = np.arange(0,500, 1)
	
	# start time
	process_startTime = time.time()

	# specify an output dir for the 'processed' results
	outputDir = join(inputDir, 'processed')
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	### Prep the gaze data ################################
	print('Prepping gaze data...')
	# format pupil data
	gazeWorld_df, frame_timestamps = formatGazeData(inputDir)
	print('n frames: %s' % frame_timestamps.shape)

	# write the gaze data (world camera coords) to a csv file
	gazeWorld_df.to_csv(join(outputDir, 'gazeData_world.csv'), sep='\t', index=False)

	# drop the invalid rows for subsequent steps
	gazeWorld_df = gazeWorld_df[gazeWorld_df.confidence > .5]

	### Prep the reference stimulus ########################
	print('Prepping reference stimulus...')
	shutil.copy(refFile, outputDir) 	# put a copy of the reference file in the outputDir
	refStim = cv2.imread(refFile)   		# load in ref stimulus
	
	refStim_dims = (refStim.shape[1], refStim.shape[0])  # pixel dims of stimulus (height, width)

	# instantiate the gazeMappingTool object
	mapper = gm.GazeMapper(cameraCalib, refStim, obj_dims)


	### Prep the video data ################################
	print('Prepping video data...')
	vid_path = join(inputDir, 'fullstream.mp4')

	# load the video, get parameters
	vid = cv2.VideoCapture(vid_path)
	if OPENCV3:
		totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
		vidSize = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		fps = vid.get(cv2.CAP_PROP_FPS)
		vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
	else:
		totalFrames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
		vidSize = (int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
		fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
		vidCodec = cv2.cv.CV_FOURCC(*'mp4v')

	# make sure no attempts on nonexistent frames
	if totalFrames > framesToUse.max():
		framesToUse = framesToUse[framesToUse <= totalFrames]

	# define output videos
	output_prefix = refFile.split('/')[-1:][0].split('.')[0] 	# set the output prefix based on the reference image
	
	vidOutFile_orig = join(outputDir, 'orig.m4v')
	vidOut_orig = cv2.VideoWriter()
	vidOut_orig.open(vidOutFile_orig, vidCodec, fps, vidSize, True)	

	vidOutFile_gaze = join(outputDir, 'gaze.m4v')
	vidOut_gaze = cv2.VideoWriter()
	vidOut_gaze.open(vidOutFile_gaze, vidCodec, fps, vidSize, True)

	vidOutFile_summaryWorld = join(outputDir, 'summaryWorld.m4v')
	vidOut_summaryWorld = cv2.VideoWriter()
	vidOut_summaryWorld.open(vidOutFile_summaryWorld, vidCodec, fps, vidSize, True)

	vidOutFile_summaryRef = join(outputDir, 'summaryRef.m4v')
	vidOut_summaryRef = cv2.VideoWriter()
	vidOut_summaryRef.open(vidOutFile_summaryRef, vidCodec, fps, refStim_dims, True)

	### Loop through frames of world video #################################
	print('Processing frames....')
	frameProcessing_startTime = time.time()
	frameCounter = 0
	while vid.isOpened():
		# read the next frame of the video
		ret, frame = vid.read()
		
		# check if it's a valid frame
		if (ret==True) and (frameCounter in framesToUse):

			# grab the gazeData (world coords) for this frame only
			thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]

			# undistort the frame
			#frame = cv2.undistort(frame, camCalib['camera_matrix'], camCalib['dist_coefs'])

			# submit this frame to the processing function
			processedFrame = processFrame(frameCounter, frame, mapper, thisFrame_gazeData_world, frame_timestamps)
			
			# if there was a good match between reference stim and world frame on this frame
			if processedFrame['foundGoodMatch']:

				# append this frames gaze data file to the bigger list
				if 'gazeData_master' in locals():
					gazeData_master = pd.concat([gazeData_master, processedFrame['gazeData']])
				else:
					gazeData_master = processedFrame['gazeData']

				# make the summary visualization (in reference stim coords)
				gazeSummary_ref = createHeatmap(gazeData_master, frameCounter, mapper.refImgColor)

				# project the gazeSummary visualization back into the world coords
				mappedSummaryFrame = mapper.projectImage2D(processedFrame['origFrame'], processedFrame['ref2frame_2Dtrans'], gazeSummary_ref)

			else:
				# otherwise, just place the original frame
				gazeSummary_ref = processedFrame['origFrame']
				mappedSummaryFrame = processedFrame['origFrame']

			# Write out this frame's different video files
			vidOut_orig.write(processedFrame['origFrame'])
			vidOut_gaze.write(processedFrame['gazeFrame'])
			vidOut_summaryRef.write(gazeSummary_ref)
			vidOut_summaryWorld.write(mappedSummaryFrame)

		# increment frame counter
		frameCounter += 1
		if frameCounter > np.max(framesToUse):
			# release all videos
			vid.release()
			vidOut_orig.release()
			vidOut_gaze.release()
			vidOut_summaryRef.release()
			vidOut_summaryWorld.release()

			# prep 3D animation
			create3Danimation(outputDir, refFile, gazeData_master)

			# write out gaze data
			try:
				gazeData_master.to_csv(join(outputDir, 'gazeData_master.tsv'), sep='\t', index=False, float_format='%.3f')
			except:
				print('cound not write gazeData_master to csv')
				pass

	endTime = time.time()
	frameProcessing_time = endTime - frameProcessing_startTime
	print('Total time: %s seconds' % frameProcessing_time)
	print('Avg time/frame: %s seconds' % (frameProcessing_time/framesToUse.shape[0]) )


def processFrame(frameCounter, frame, mapper, thisFrame_gazeData_world, frame_timestamps):
	""" Compute all transformations on a given frame """

	fr = {}		# create dict to store info for this frame
	fr['frameNum'] = frameCounter		# store frame number
	
	# create copy of original frame
	origFrame = frame.copy()
	fr['origFrame'] = origFrame 		# store

	# convert to grayscale
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# get the timestamp for this frame
	frame_ts = frame_timestamps[frameCounter]
	fr['frame_ts'] = frame_ts 			# store

	# Try to process this frame
	try:
		# find the key points and features on this frame
		frame_kp, frame_des = mapper.findFeatures(frame_gray)
		print('found %s features on frame %s' %(len(frame_kp), frameCounter))

		# look for matching keypoints on the reference stimulus
		if len(frame_kp) < 2:
			ref_matchPts = None
		else:
			ref_matchPts, frame_matchPts = mapper.findMatches(frame_gray, frame_kp, frame_des)

		# check if matches were found
		try:		
			numMatches = ref_matchPts.shape[0]

			# if sufficient number of matches....
			if numMatches > 10:
				print('found %s matches on frame %s' %(numMatches, frameCounter))
				sufficientMatches = True
			else:
				print('Insufficient matches (%s matches) on frame %s' %(numMatches, frameCounter))
				sufficientMatches = False

		except:
			print ('no matches found on frame %s' % frameCounter)
			sufficientMatches = False
			pass

		fr['foundGoodMatch'] = sufficientMatches
		
		# Uses matches to find 2D and 3D transformations
		if sufficientMatches:
			### 3D operations ##########################
			# get mapping from camera to 3D location of reference image. Reference match points treated as 2D plane in the world (z=0)
			rvec, tvec = mapper.PnP_3Dmapping(ref_matchPts, frame_matchPts)

			# calculate camera position & orientation
			camPosition, camOrientation = mapper.getCameraPosition(rvec, tvec)
			fr['camPosition'] = camPosition
			fr['camOrientation'] = camOrientation

			### 2D operations ###########################
			# get the transformation matrices to map between world frame and reference stimuli
			ref2frame_2D, frame2ref_2D = mapper.get2Dmapping(ref_matchPts, frame_matchPts)
			fr['ref2frame_2Dtrans'] = ref2frame_2D
			fr['frame2ref_2Dtrans'] = frame2ref_2D

			### Gaze data operations ####################
			if thisFrame_gazeData_world.shape[0] == 0:
				# if no gaze points for this frame
				drawGazePt = False

				# store empty dataframe to store gaze data in frame, reference, and object coordinates
				gazeData_df = pd.DataFrame(columns=['gaze_ts', 'worldFrame', 'confidence',
											'frame_gazeX', 'frame_gazeY',
											'ref_gazeX', 'ref_gazeY', 
											'obj_gazeX', 'obj_gazeY', 'obj_gazeZ', 
											'camX', 'camY', 'camZ',
											'camTheta', 'camRX', 'camRY', 'camRZ'])
				fr['gazeData'] = gazeData_df
				fr['gazeFrame'] = origFrame

			else:
				drawGazePt = True

				# create dataframes to write gaze data into
				gazeData_df = pd.DataFrame(columns=['gaze_ts', 'worldFrame', 'confidence',
											'frame_gazeX', 'frame_gazeY',
											'ref_gazeX', 'ref_gazeY', 
											'obj_gazeX', 'obj_gazeY', 'obj_gazeZ',
											'camX', 'camY', 'camZ',
											'camTheta', 'camRX', 'camRY', 'camRZ'])
				
				# get the camera position and orientation
				camX = camPosition[0]
				camY = camPosition[1]
				camZ = camPosition[2]
				camTheta = camOrientation[0]
				camRX = camOrientation[1]
				camRY = camOrientation[2]
				camRZ = camOrientation[3]
				
				# grab all gaze data for this frame, translate to different coordinate systems
				for i,gazeRow in thisFrame_gazeData_world.iterrows():
					ts = gazeRow['timestamp']
					frameNum = frameCounter
					conf = gazeRow['confidence']

					# translate normalized gaze location to screen coords
					frame_gazeX = gazeRow['norm_pos_x'] * frame_gray.shape[1]
					frame_gazeY = gazeRow['norm_pos_y'] * frame_gray.shape[0]

					# convert coordinates from frame to reference stimulus coordinates
					ref_gazeX, ref_gazeY = mapper.mapCoords2D((frame_gazeX, frame_gazeY), frame2ref_2D)

					# convert from reference stimulus to object coordinates
					objCoords = mapper.ref2obj(np.array([ref_gazeX, ref_gazeY]).reshape(1,2))
					obj_gazeX, obj_gazeY, obj_gazeZ = objCoords.ravel()
 
					# create dict
					thisRow_df = pd.DataFrame({'gaze_ts': ts, 'worldFrame': frameNum, 'confidence':conf,
												'frame_gazeX': frame_gazeX, 'frame_gazeY': frame_gazeY,
												'ref_gazeX': ref_gazeX, 'ref_gazeY': ref_gazeY,
												'obj_gazeX': obj_gazeX, 'obj_gazeY': obj_gazeY, 'obj_gazeZ': obj_gazeZ,
												'camX': camX, 'camY': camY, 'camZ': camZ,
												'camTheta': camTheta, 'camRX':camRX, 'camRY':camRY, 'camRZ':camRZ},
												index=[i])

					# append this row to the gaze data dataframe
					gazeData_df = pd.concat([gazeData_df, thisRow_df])

				# store gaze data
				fr['gazeData'] = gazeData_df

				# draw circles for gaze locations
				gazeFrame = origFrame.copy()
				for i,row in gazeData_df.iterrows():
					frame_gazeX = int(row['frame_gazeX'])
					frame_gazeY = int(row['frame_gazeY'])

					# set color for last value to be different than previous values for this frame
					if i == gazeData_df.index.max():
						cv2.circle(gazeFrame, (frame_gazeX, frame_gazeY), 10, [96, 52, 234], -1)
					else:
						cv2.circle(gazeFrame, (frame_gazeX, frame_gazeY), 8, [168, 231, 86], -1)

				# store the gaze frame
				fr['gazeFrame'] = gazeFrame
			
		elif not sufficientMatches:
			fr['gazeFrame'] = origFrame	
	
	except:
		print('gaze mapping failed on frame {}'.format(frameCounter))
		fr['gazeFrame'] = origFrame	
		fr['foundGoodMatch'] = False	

	# Return the dict holding all of the info for this frame
	return fr


def createHeatmap(gazeData_master, frameCounter, refStim):
	"""
	Create a heatmap base on x and y gaze values that have been mapped to the reference stimulus
	Return a cv2 img of the heatmap that is sized according to the refStim
	"""
	# retrieve the gaze values mapped to the reference stim
	heatmap_df = gazeData_master.loc[gazeData_master['worldFrame'] <= frameCounter, ['worldFrame', 'ref_gazeX', 'ref_gazeY']]

	# start the plot
	refStim_RGB = cv2.cvtColor(refStim, cv2.COLOR_BGR2RGB)
	w = refStim.shape[1]
	h = refStim.shape[0]
	hmDPI = 150
	fig = plt.figure(figsize=(w/hmDPI, h/hmDPI), dpi=hmDPI)
	ax = plt.axes([0,0,1,1])
	ax.imshow(refStim_RGB)

	# draw the heatmap (if there's enough data)
	if heatmap_df.shape[0] <= 2:
		heatmap = refStim
	else:
		xArr = heatmap_df['ref_gazeX'].values
		yArr = heatmap_df['ref_gazeY'].values

		# use Seaborne to create a pretty heatmap (as kernel density estimate)
		sns.set_style('white')
		sns.despine(trim=True)
		ax = sns.kdeplot(xArr, yArr,
						shade=True,
						shade_lowest=False,
						cmap='viridis',
						alpha=0.7)
		ax.set_xlim(0,w)
		ax.set_ylim(0,h)
		plt.axis('off')
		plt.gca().invert_yaxis()

		# need to draw in order to access the pixel data
		fig.canvas.draw()

		# store figure as np array
		heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		heatmap = heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		plt.close(fig)

		# add gaze trace
		for i,row in heatmap_df.iterrows():
			if i == heatmap_df.index.min():
				# set the starting point
				prev_ref_gazeX = int(row['ref_gazeX'])
				prev_ref_gazeY = int(row['ref_gazeY'])
			else:
				cur_ref_gazeX = int(row['ref_gazeX'])
				cur_ref_gazeY = int(row['ref_gazeY'])

				# draw line connecting previous point to current
				cv2.line(heatmap, (prev_ref_gazeX, prev_ref_gazeY), (cur_ref_gazeX, cur_ref_gazeY), [107, 234, 101], 3, cv2.LINE_AA)

				# update previous point
				prev_ref_gazeX = cur_ref_gazeX
				prev_ref_gazeY = cur_ref_gazeY

		# add circles for the last dots
		for i,row in heatmap_df.loc[heatmap_df['worldFrame'] == frameCounter, :].iterrows():
			ref_gazeX = int(row['ref_gazeX'])
			ref_gazeY = int(row['ref_gazeY'])

			# set color for last value to be different than previous values for this frame
			if i == heatmap_df.index.max():
				cv2.circle(heatmap, (ref_gazeX, ref_gazeY), 10, [234, 52, 96], -1)
			else:
				cv2.circle(heatmap, (ref_gazeX, ref_gazeY), 8, [86, 231, 168], -1)
		
		# convert to rgb
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

	return heatmap


def create3Danimation(output_dir, referenceImage_file, gazeData_df):
	"""
	create an animated 3D reconstruction of viewing behavior
	based on the javascript template
	"""

	# copy the template directory
	if os.path.isdir(join(output_dir, '3D')):	
		shutil.rmtree(join(output_dir, '3D'))			# remove if already exists
	shutil.copytree('visTemplate_3D', join(output_dir, '3D'))

	# replace obj dimension vars in javascript file with the dims specified here
	for line in fileinput.input(join(output_dir, '3D', 'gazeMapping3D.js'), inplace=True):
		if 'var targetWidth =' in line:
			line = line.replace('var targetWidth = 76;', 'var targetWidth = {};'.format(str(obj_dims[1])))
		if 'var targetHeight =' in line:
			line = line.replace('var targetHeight = 82;', 'var targetHeight = {};'.format(str(obj_dims[0])))
		sys.stdout.write(line) 

	# make resized copy of the reference image
	refStim = cv2.imread(referenceImage_file)   		# load in ref stimulus
	im_h = refStim.shape[0]
	im_w = refStim.shape[1]
	aspect_ratio = im_h/im_w
	new_w = 600   										# max width
	new_h = int(new_w * aspect_ratio)
	refStim_small = cv2.resize(refStim, (new_w, new_h))
	cv2.imwrite(join(output_dir, '3D', 'textures', 'target_small.jpg'), refStim_small)

	### format the gaze_df
	# there are multiple gaze values per frame; take the average by frame
	df_byFrame = gazeData_df.groupby('worldFrame').mean()

	# interpolate over any missing values
	df_byFrame.interpolate(method='linear', inplace=True)

	# apply smoothing function the the camera position/orientation columns
	dfSmooth = df_byFrame.loc[:, ['camX', 'camY', 'camZ', 'camTheta', 'camRX', 'camRY', 'camRZ']].apply(smoothMotion, axis=0)

	# combine the smoothed camera movements with the gaze coordinates (in 3D object space)
	animation_df = pd.concat([dfSmooth, df_byFrame.loc[:, ['obj_gazeX', 'obj_gazeY', 'obj_gazeZ']]], axis=1)

	# write to csv file
	animation_df.to_csv(join(output_dir, '3D', 'data', 'camera_and_gaze_smooth.csv'), float_format='%.3f', sep=',', index=False)


def smoothMotion(arr):
	"""
	Apply a Gaussian filter to the supplied array
	"""
	keepGoing = True
	while keepGoing:
		# calculate the 1st order differential
		arrDiff = np.diff(arr)
		arrDiff = np.insert(arrDiff, 0, 0)		# insert 0 to make matching arr size

		# remove datapts where the difference is greater than 5 and interpolate over
		arrFilt = arr.copy()
		arrFilt[np.abs(arrDiff)>3] = np.nan

		if sum(np.isnan(arrFilt)) == 0:
		    keepGoing = False

		arrInterp = pd.Series(arrFilt).interpolate()

		# smooth with Gaussian filter
		arr = filters.gaussian_filter1d(arrInterp, 3)

	return arr


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
		print('Input data copied to: {}'.format(newDataDir))

		# processing the data (note: the output dir from previous step is now the input dir for this step)
		processRecording(newDataDir, args.referenceFile, args.cameraCalibration)




