import glob
from timeit import default_timer as timer
import cv2 as cv
from Camera import Camera
from Detector import Detector
from config import *
from util import displayTextOnImage


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
    detectObjects = True

    if videoCapture.isOpened():
        print("Playback started!")

        # Get the maximum framerate of the video
        videoFramerate = int(videoCapture.get(cv.CAP_PROP_FPS))
        frameRates = []

        while videoCapture.isOpened() and valid:
            valid, frame = videoCapture.read()

            if valid:
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
                if detectObjects:
                    car_overlay = detector.detectCars(frame)
                    sign_overlay = detector.detectSigns(frame)

                    object_overlay = cv.bitwise_or(car_overlay, sign_overlay)
                    lane_frame = cv.addWeighted(lane_frame, 1, object_overlay, 0.5, 1)

                # Undistort the image
                lane_frame = camera.undistort(lane_frame)

                # Calculate the Framerate
                frameTime = timer() - startTimer
                frameRate = int(1 / frameTime)
                frameRates.append(frameRate)
                displayTextOnImage(lane_frame, f"FPS: {frameRate} / {videoFramerate}", (5, 15))

                # Display the line curvature
                curvature = detector.getCurvature()
                if curvature is not None:
                    curvatureText = f"Curvature: {round(curvature, 0)} m"
                else:
                    curvatureText = "No lines detected!"
                displayTextOnImage(lane_frame, curvatureText, (5, 35))
                displayTextOnImage(lane_frame, f"Car offset: {detector.getCarPosition()} m", (3, 55))

                # Show the video feeds
                cv.imshow("Video Playback", lane_frame)
                cv.imshow("Birds-Eye", birdsEye)

                # Check if escape key or 'q' was pressed
                key = cv.waitKey(5)
                if key == 27 or key == ord("q"):
                    print("ESC or Q pressed! Exiting playback!")
                    valid = False
                elif key == ord("p"):
                    cv.waitKey(0)

        print(f"Video finished playing with a mean framerate of {np.around(np.mean(frameRates), 1)}")
    else:
        print("Couldn't start the playback!")

    # Release the video file and close all windows
    videoCapture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
