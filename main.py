import glob

import cv2 as cv
from Camera import Camera


def main():
    camera = Camera()
    projectionError = camera.calibrate(glob.glob('img/Udacity/calib/*.jpg'))
    print(f"Camera calibrated with re-projection error of {projectionError}")

    distorted = cv.imread("img/Udacity/image001.jpg")
    cv.imshow("Distorted", distorted)
    cv.imshow("Undistorted", camera.undistort(distorted))
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
