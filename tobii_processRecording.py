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

OPENCV3 = (cv2.__version__.split('.')[0] == '3')
print("OPENCV version " + cv2.__version__)
