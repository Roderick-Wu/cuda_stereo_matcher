import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from stereo_disparity_score import stereo_disparity_score
from algorithm import stereo_disparity_best

import time

# Load the stereo images and ground truth.
# Il = imread("../images/cones/cones_image_02.png", as_gray = True)
# Ir = imread("../images/cones/cones_image_06.png", as_gray = True)
Il = imread("../images/cones/cones_image_02.png", mode='F')
Ir = imread("../images/cones/cones_image_06.png", mode='F')
#Il = imread("../images/teddy/teddy_image_02.png", mode='F')
#Ir = imread("../images/teddy/teddy_image_06.png", mode='F')


# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
# It = imread("../images/cones/cones_disp_02.png",  as_gray = True)/4.0
It = imread("../images/cones/cones_disp_02.png",  mode='F')/4.0
#It = imread("../images/teddy/teddy_disp_02.png",  mode='F')/4.0

# Load the appropriate bounding box.
bbox = np.load("../data/cones_02_bounds.npy")

time_start = time.time()

Id = stereo_disparity_best(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)

print("Time taken: %.2f seconds" % (time.time() - time_start))
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))





Il = imread("../images/teddy/teddy_image_02.png", mode='F')
Ir = imread("../images/teddy/teddy_image_06.png", mode='F')


It = imread("../images/teddy/teddy_disp_02.png",  mode='F')/4.0


# Load the appropriate bounding box.
bbox = np.load("../data/teddy_02_bounds.npy")

time_start = time.time()

Id = stereo_disparity_best(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)

print("Time taken: %.2f seconds" % (time.time() - time_start))
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))



plt.imshow(Id, cmap = "gray")
plt.show()