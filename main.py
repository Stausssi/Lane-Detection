import glob

import cv2 as cv
from Camera import Camera
from Detector import Detector


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

    # Whether other objects (cars, etc.) should be detected
    detectObjects = True

    if videoCapture.isOpened():
        print("Playback started!")

        while videoCapture.isOpened() and valid:
            valid, frame = videoCapture.read()

            if valid:
                # Undistort the image
                frame = camera.undistort(frame)

                # Now preprocess the image (segmentation, color filtering, etc.)

                # Detect the lines
                # frame = Detector.detectLines(frame)
                cv.imshow("Birds-Eye", camera.birdsEyeView(frame))

                # Also detect objects, if wanted
                if detectObjects:
                    frame = Detector.detectObjects(frame)

                # Show the image
                cv.imshow("Video Playback", frame)

                # Check if escape key or 'q' was pressed
                key = cv.waitKey(10)
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
