# All videos are 1280x720, so just hardcode this size
import numpy as np

IMAGE_SIZE = (1280, 720)
WIDTH, HEIGHT = IMAGE_SIZE

# Debugging
DRAW_HOUGH = True

# The ROI of the lane detection and camera warp
CAR_HOOD_HEIGHT = 60
PADDING = 150
ROI = [
    [.55 * WIDTH, 0.63 * HEIGHT],  # Top right
    [WIDTH - PADDING, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom right
    [PADDING, HEIGHT - CAR_HOOD_HEIGHT],  # Bottom left
    [.45 * WIDTH, 0.63 * HEIGHT],  # Top left
]

# The destination ROI of the birds eye view
WARPED_ROI = [
    [WIDTH - PADDING, 0],  # Top right
    [WIDTH - PADDING, HEIGHT],  # Bottom right
    [PADDING, HEIGHT],  # bottom left
    [PADDING, 0],  # Top left
]

LINE_TOLERANCE = 75
MAX_LIFETIME = 20

Y_RANGE = np.arange(ROI[0][1], HEIGHT)

LANE_COLOR = (0, 128, 0)
