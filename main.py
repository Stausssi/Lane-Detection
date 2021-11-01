import glob

import cv2 as cv
from Camera import Camera


def main():
    print("Calibrating camera...")
    camera = Camera()
    projectionError = camera.calibrate(glob.glob('img/Udacity/calib/*.jpg'))
    print(f"Camera calibrated with re-projection error of {projectionError}!")

    print("Starting video playback...")

    # Read the video
    videoCapture = cv.VideoCapture("img/Udacity/project_video.mp4")
    valid = True

    if videoCapture.isOpened():
        print("Playback started!")

        while videoCapture.isOpened() and valid:
            valid, frame = videoCapture.read()

            if valid:
                # Undistort the image
                frame = camera.undistort(frame)

                # Show the image
                cv.imshow("Video Playback", frame)

                # Check if escape key was pressed
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
