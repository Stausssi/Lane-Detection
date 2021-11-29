# All videos are 1280x720, so just hardcode this size
imageSize = (1280, 720)
width, height = imageSize

# The ROI of the lane detection and camera warp
carHoodHeight = 60
padding = 150
roi = [
    [.55 * width, 0.63 * height],  # Top right
    [width - padding, height - carHoodHeight],  # Bottom right
    [padding, height - carHoodHeight],  # Bottom left
    [.45 * width, 0.63 * height],  # Top left
]

# The destination ROI of the birds eye view
warpedROI = [
    [width - padding, 0],  # Top right
    [width - padding, height],  # Bottom right
    [padding, height],  # bottom left
    [padding, 0],  # Top left
]

lineTolerance = 15
maximumLifetime = 10
