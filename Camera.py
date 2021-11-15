import os

import cv2 as cv
import numpy as np


class Camera:
    def __init__(self):
        self.matrix = None
        self.distortion = None
        self.rotationVectors = None
        self.translationVectors = None

    def calibrate(self, images, boardSize=(6, 9), show=False) -> float:
        """
        This method calibrates the camera and sets the internal values.

        Args:
            images (list[str]): A list of strings, which represent a path to a calibration image.
            boardSize (tuple[int, int]): The dimensions of the chessboard.
            show (bool): Whether the found chessboard corners should be shown.

        Returns:
            float: The overall RMS re-projection error. -1, if no images where given
        """

        if len(images) > 0:
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Prepare object points. A Calibration picture consists of X,Y fields defined by the boardSize
            objects = np.zeros((boardSize[1] * boardSize[0], 3), np.float32)
            objects[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)

            # Save object and image points
            objectPoints = []
            imagePoints = []

            gray = None

            # Go over every calibration file
            for index, file in enumerate(images):
                # print(file)
                img = cv.imread(file, cv.COLOR_BGR2RGB)
                gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

                # Let OpenCV find the corners
                projectionError, corners = cv.findChessboardCorners(gray, boardSize)

                # if any corners where found, save image and objects
                if projectionError:
                    objectPoints.append(objects)

                    cornerSub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imagePoints.append(corners)

                    # Display the found corners
                    cv.drawChessboardCorners(img, boardSize, cornerSub, projectionError)

                    if show:
                        cv.imshow(f"Chessboard corners", img)
                        cv.waitKey(500)
                else:
                    print(f"[Camera calibration]: No corners found in '{os.path.basename(file)}'!")

            if show:
                cv.destroyAllWindows()

            projectionError, self.matrix, self.distortion, self.rotationVectors, self.translationVectors = cv.calibrateCamera(
                objectPoints,
                imagePoints,
                gray.shape[::-1],
                None,
                None
            )

            return projectionError
        else:
            return -1

    def undistort(self, img, alpha=1):
        """
        This method undistorts a given image.

        Args:
            img (np.ndarray): The image to undistort.
            alpha (int): The alpha of the optimal camera matrix

        Returns:
            img: The undistorted image.
        """

        # The OpenCV method returns the new matrix and a ROI, which will be unpacked
        newMatrix, (x, y, h, w) = cv.getOptimalNewCameraMatrix(self.matrix, self.distortion, img.shape[:2], alpha)

        # Undistort and crop the image to the ROI
        return cv.undistort(img, self.matrix, self.distortion, None, newMatrix)[y:h + y // 4, x:w - x // 4]

    @staticmethod
    def birdsEyeView(img):
        """
        This method transform an image into the birds-eye view.

        Args:
            img (np.ndarray): The image to transform

        Returns:
            img: An image in birds eye view
        """

        # Grab the previous resolution
        if len(img.shape) > 2:
            height, width, _ = img.shape
        else:
            height, width = img.shape

        # Resize the rectangles to fit the image size
        img = cv.resize(img, (1280, 720))

        # Create the rectangles (copied from the script)
        src_rect = np.float32([
            [278, 668], [1026, 668],
            [598, 448], [684, 448]
        ])
        dst_rect = np.float32([
            [300, 720], [980, 720],
            [300, 0], [980, 0]
        ])

        # Create the transformation matrix
        matrix = cv.getPerspectiveTransform(src_rect, dst_rect)

        return cv.resize(cv.warpPerspective(img, matrix, (1280, 720)), (width, height))
