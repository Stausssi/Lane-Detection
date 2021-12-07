import glob

import cv2 as cv
from timeit import default_timer as timer

from Camera import Camera
from Detector import Detector
from util import displayTextOnImage
from config import *


def main():
    print("Calibrating camera...")
    camera = Camera()
    projectionError = camera.calibrate(glob.glob('img/Udacity/calib/*.jpg'))
    print(f"Camera calibrated with re-projection error of {projectionError}!")

    print("Starting video playback...")

    # Read the video
    videoCapture = cv.VideoCapture("img/Udacity/project_video.mp4")
    # videoCapture = cv.VideoCapture("img/Udacity/challenge_video.mp4")
    # videoCapture = cv.VideoCapture("img/Udacity/harder_challenge_video.mp4")
    valid = True

    detector = Detector()

    # Whether other objects (cars, etc.) should be detected
    detectObjects = False

    if videoCapture.isOpened():
        print("Playback started!")

        # Get the maximum framerate of the video
        videoFramerate = int(videoCapture.get(cv.CAP_PROP_FPS))

        while videoCapture.isOpened() and valid:
            valid, frame = videoCapture.read()

            if valid:
                # Start the timer
                startTimer = timer()

                # Birds-Eye view
                birdsEye = camera.birdsEyeView(frame)

                # Detect the lines
                lane_overlay = detector.detectLines(frame)

                # Overlay the lane overlay on the image
                lane_frame = cv.addWeighted(frame, 1, lane_overlay, 0.5, 1)

                # Also detect objects, if wanted
                if detectObjects:
                    object_overlay = detector.detectObjects(frame)

                # Undistort the image
                lane_frame = camera.undistort(lane_frame)

                # Calculate the Framerate
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

                # Show the video feeds
                cv.imshow("Video Playback", lane_frame)
                cv.imshow("Birds-Eye", birdsEye)
                # cv.imshow("Lines", lane_overlay)
                # cv.imshow("Lines Birds-Eye", detector.detectLines(birdsEye, False))

                # Check if escape key or 'q' was pressed
                key = cv.waitKey(5)
                if key == 27 or key == ord("q"):
                    print("ESC or Q pressed! Exiting playback!")
                    valid = False

        print("Video finished!")
    else:
        print("Couldn't start the playback!")

    # Release the video file and close all windows
    videoCapture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
