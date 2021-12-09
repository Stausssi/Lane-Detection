import numpy as np
import cv2

# All videos are 1280x720, so just hardcode this size
IMAGE_SIZE = (1280, 720)
WIDTH, HEIGHT = IMAGE_SIZE

# Debugging
# Draw an HSV histogram. ATTENTION!! This is very heavy on the performance.
SHOW_HIST = False

# Show the filtered image
SHOW_FILTERED = True

# Draw the lines of the Hough detection
DRAW_HOUGH = False

# The ROI of the lane detection and camera warp
CAR_HOOD_HEIGHT = 60
PADDING = 100
ROI = [
    [.555 * WIDTH, 0.63 * HEIGHT],  # Top right
    [WIDTH - PADDING, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom right
    [PADDING * 2, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom left
    [.45 * WIDTH, 0.63 * HEIGHT],  # Top left
]

# The destination ROI of the birds eye view
WARPED_ROI = [
    [WIDTH - PADDING, 0],  # Top right
    [WIDTH - PADDING, HEIGHT],  # Bottom right
    [PADDING, HEIGHT],  # bottom left
    [PADDING, 0],  # Top left
]

# The values control how often a line is changed
LINE_TOLERANCE = 17
MAX_LIFETIME = 10

# The range to calculate the fitted polynoms in
Y_RANGE = np.arange(0, HEIGHT)

# For Curvature and Center offset
# MPP = meters per pixel
MPP_Y = 30 / HEIGHT
MPP_X = 4 / WIDTH

# The height where the evaluation of the polynom should take place
EVALUATION_Y = HEIGHT - CAR_HOOD_HEIGHT

DEFAULT_CENTER = WIDTH // 2

# Line Color Filters
MIN_YELLOW = np.array([15, 75, 200], dtype="uint8")
MAX_YELLOW = np.array([25, 255, 255], dtype="uint8")

MIN_WHITE = np.array([0, 0, 220], dtype="uint8")
MAX_WHITE = np.array([255, 35, 255], dtype="uint8")

# The color to draw the detected line in
LANE_COLOR = (0, 128, 0)

# Car Detection
CARS_CASCADE = cv2.CascadeClassifier('cars.xml')

# Road Sign Detection
MIN_GREEN_SIGN = np.array([70, 75, 35], dtype="uint8")
MAX_GREEN_SIGN = np.array([85, 255, 255], dtype="uint8")

MIN_YELLOW_SIGN = np.array([10, 185, 100], dtype="uint8")
MAX_YELLOW_SIGN = np.array([25, 255, 200], dtype="uint8")
