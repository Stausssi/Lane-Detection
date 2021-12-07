import cv2 as cv
import numpy as np

# All videos are 1280x720, so just hardcode this size
IMAGE_SIZE = (1280, 720)
WIDTH, HEIGHT = IMAGE_SIZE

# Debugging
# Draw an HSV histogram. ATTENTION!! This is very heavy on the performance.
SHOW_HIST = False

# Show the color and edge filtered image
SHOW_COMBINED = True

# Draw the lines of the Hough detection
DRAW_HOUGH = False

# The ROI of the lane detection and camera warp
CAR_HOOD_HEIGHT = 60
PADDING = 150
ROI = [
    [.55 * WIDTH, 0.63 * HEIGHT],  # Top right
    [WIDTH - PADDING, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom right
    [PADDING, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom left
    [.45 * WIDTH, 0.63 * HEIGHT],  # Top left
]

IGNORED_ROI = [
    [ROI[0][0] * 1.1, ROI[0][1] * 1.2],  # Top right
    [ROI[1][0] * 0.8, ROI[1][1]],  # Bottom right
    [ROI[2][0] * 3, ROI[2][1]],  # Bottom left
    [ROI[3][0], ROI[3][1] * 1.2]  # Top left
]

# The destination ROI of the birds eye view
WARPED_ROI = [
    [WIDTH - PADDING, 0],  # Top right
    [WIDTH - PADDING, HEIGHT],  # Bottom right
    [PADDING, HEIGHT],  # bottom left
    [PADDING, 0],  # Top left
]

# The values control how often a line is changed
LINE_TOLERANCE = 50
MAX_LIFETIME = 15

# The range to calculate the fitted polynoms in
Y_RANGE = np.arange(ROI[0][1], HEIGHT)

# Line Color Filters
MIN_YELLOW = np.array([15, 100, 100], dtype="uint8")
MAX_YELLOW = np.array([100, 255, 255], dtype="uint8")

MIN_WHITE = np.array([0, 0, 220], dtype="uint8")
MAX_WHITE = np.array([255, 50, 255], dtype="uint8")

CROSS_FILTER_5_5 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

# The color to draw the detected line in
LANE_COLOR = (0, 128, 0)

# Car Detection
CARS_CASCADE = cv.CascadeClassifier('cars.xml')
