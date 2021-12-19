import glob
from timeit import default_timer as timer
import cv2 as cv
import numpy as np

from Camera import Camera
from Detector import Detector
from config import DEFAULT_ROI, KITTI_ROI
from util import displayTextOnImage


def detectionPipeline(
        detector, camera, frame, detectSigns, videoFramerate=-1, keyDelay=1, showBirdseye=False, isKITTI=False
):
    """
    The main pipeline an image will go through to detect lines.

    Args:
        detector (Detector): The detector object
        camera (Camera): The calibrated camera object
        frame (np.ndarray): the image to process
        detectSigns (bool): Whether signs should be detected
        videoFramerate (int): The framerate of the video-playback (if any)
        keyDelay (int): The delay cv.waitKey() should wait for. 0 means indefinite
        showBirdseye (bool): Whether the birds-eye perspective should be shown
        isKITTI (bool): Whether the image is from KITTI -> No undistort

    Returns:
        tuple[bool, int]: A tuple containing whether the playback should be exited and the current frame rate
    """

    startTimer = None
    if videoFramerate > 0:
        # Start the timer
        startTimer = timer()

    # Birds-Eye view
    birdsEye = camera.birdsEyeView(frame)

    # Detect the lines
    lane_overlay = detector.detectLines(birdsEye)

    # Re-transform the overlay from birds eye
    lane_overlay = camera.normalView(lane_overlay)

    # Overlay the lane overlay on the image
    lane_frame = cv.addWeighted(frame, 1, lane_overlay, 0.5, 1)

    # Also detect objects, if wanted
    if detectSigns:
        sign_overlay = detector.detectSigns(frame)

        lane_frame = cv.bitwise_or(lane_frame, sign_overlay)

    # Undistort the image
    if not isKITTI:
        lane_frame = camera.undistort(lane_frame)

    # Calculate the Framerate if this is a video playback
    frameRate = -1
    if videoFramerate > 0:
        # Grab the time the frame took
        frameTime = timer() - startTimer
        frameRate = int(1 / frameTime)
        displayTextOnImage(lane_frame, f"FPS: {frameRate} / {videoFramerate}", (5, 15))

    # Display the line curvature
    curvature = detector.getCurvature()
    if curvature is not None:
        curvatureText = f"Curvature: {round(curvature, 0)} m"
    else:
        curvatureText = "No lines detected!"
    displayTextOnImage(lane_frame, curvatureText, (5, 35))
    displayTextOnImage(lane_frame, f"Center offset: {detector.getCenterOffset()} m", (3, 55))

    # Show the output image
    cv.imshow("Lane Detection", lane_frame)
    if showBirdseye:
        cv.imshow("Birds-Eye", birdsEye)

    # Check if escape key or 'q' was pressed
    key = cv.waitKey(keyDelay)
    if key == 27 or key == ord("q"):
        print("ESC or Q pressed! Exiting!")
        return False, frameRate
    elif key == ord("p") and videoFramerate > 0:
        cv.waitKey(0)

    return True, frameRate


def main():
    print("Calibrating camera...")
    camera = Camera()
    projectionError = camera.calibrate(glob.glob('img/Udacity/calib/*.jpg'))
    print(f"Camera calibrated with re-projection error of {projectionError}!")

    # Whether the video playback should be shown
    useVideo = True

    # Whether the Udacity images should be shown
    useImages = True

    # Whether the KITTI images should be shown
    # KiTTI should use a different perspective transform
    # Also, arrows and other line markings should be filtered
    useKITTI = True

    # Whether sign should be detected
    detectSigns = False

    if useVideo:
        print("Starting video playback...")

        # Read the video
        videoCapture = cv.VideoCapture("img/Udacity/project_video.mp4")
        valid = True

        detector = Detector()

        if videoCapture.isOpened():
            print("Playback started!")

            # Get the maximum framerate of the video
            videoFramerate = int(videoCapture.get(cv.CAP_PROP_FPS))
            frameRates = []

            while videoCapture.isOpened() and valid:
                valid, frame = videoCapture.read()

                if valid:
                    valid, frameRate = detectionPipeline(detector, camera, frame, detectSigns, videoFramerate)
                    frameRates.append(frameRate)

            print(f"Video finished playing with a mean framerate of {np.around(np.mean(frameRates), 1)}.")
        else:
            print("Couldn't start the playback!")

        # Release the video file and close all windows
        videoCapture.release()

    if useImages or useKITTI:
        print("Starting single image detection")

        # Import the wanted images
        imageFiles = []
        if useImages:
            imageFiles.extend(glob.glob('img/Udacity/*.jpg', recursive=False))

        if useKITTI:
            imageFiles.extend(glob.glob('img/KITTI/*.jpg', recursive=False))

        # Go through every image
        for file in imageFiles:
            print(f"Showing {file}...")

            # Change ROI for KITTI
            isKITTI = "KITTI" in file
            if isKITTI:
                camera.updateROI(KITTI_ROI)
            else:
                camera.updateROI(DEFAULT_ROI)

            # Create a new detector to reset previous lanes
            detector = Detector()

            image = cv.imread(file)

            if not detectionPipeline(detector, camera, image, True, keyDelay=0, isKITTI=isKITTI)[0]:
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
