import cv2 as cv
import numpy as np
from typing import List, Tuple
from config import *


class Detector:
    def __init__(self):
        self.polyPoints = {
            roi[0][0]: [roi[0][1], maximumLifetime],
            roi[1][0]: [roi[1][1], maximumLifetime],
            roi[2][0]: [roi[2][1], maximumLifetime],
            roi[3][0]: [roi[3][1], maximumLifetime]
        }

    # ---------- [Lane Detection] ---------- #

    def detectLines(self, img, segment=True):
        """
        This method detects the lines in the given image. This task is split into subtasks.

        Args:
            img (np.ndarray): The image to detect lines on
            segment (bool): Whether the picture should be segmented

        Returns:
            np.ndarray: The image with lines drawn onto it
        """

        # TODO: Idea: segment, then filter by color, then canny -> No other lines distracting the detection
        # Use filtering (by shape and color)
        lineShapes = self._filterLineShape(img)
        color = self._filterColor(img)

        # First, segment the image
        if segment:
            lineShapes = self._segmentImage(lineShapes)
            color = self._segmentImage(color)

        combined = cv.addWeighted(lineShapes, 1, color, 1, 0)
        houghImg = combined.copy()
        combined = cv.cvtColor(combined, cv.COLOR_GRAY2RGB)
        # Performance variant
        lines = cv.HoughLinesP(houghImg, 1, np.pi / 180, 50, maxLineGap=50)

        # This is a list containing a dict for the left and right lane mapping x to y values
        lanes = [{}, {}]

        if lines is not None:
            # validLines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

            # Go over every hough line
            for line in lines:
                line = line[0]
                x1, y1 = (line[0], line[1])
                x2, y2 = (line[2], line[3])

                if abs(y1 - y2) > 25:
                    # Calculate the angle of the line to the image
                    lineAngle = cv.fastAtan2(y2 - y1, x2 - x1)

                    # Select the dictionary depending on the angle
                    lanes[int(lineAngle > 270)].update({
                        x1: y1
                    })

                    lanes[int(lineAngle > 270)].update({
                        x2: y2
                    })

            # Prepare the dictionary containing the 4 points needed for the polygon
            polyPoints = {
                roi[0][0]: -1,
                roi[1][0]: -1,
                roi[2][0]: -1,
                roi[3][0]: -1
            }

            for index, lane in enumerate(lanes):
                if len(lane) > 0:
                    # TODO: maybe train on y data
                    # Estimate the polynomial function
                    fittedPoly = np.polyfit(list(lane.keys()), list(lane.values()), 1)
                    estimator = np.poly1d(fittedPoly)

                    # Values range from the bottom left/right to the top left/right corner respectively
                    bottom = roi[index * 2][0]
                    top = roi[index * 2 + 1][0]

                    polyPoints.update({
                        bottom: estimator(bottom),
                        top: estimator(top)
                    })

                    range_x = np.arange(bottom, top)
                    cv.polylines(
                        combined,
                        [np.int32(np.asarray([range_x, estimator(range_x)]).T)],
                        False,
                        color=(0, 255, 255),
                        thickness=5
                    )

            for key, value in polyPoints.items():
                oldValue = self.polyPoints.get(key)[0]
                if abs(value - oldValue) < lineTolerance or self.polyPoints.get(key)[1] >= maximumLifetime:
                    # print(f"Changed point from {self.polyPoints[index]} to {point}.")
                    self.polyPoints.update({
                        key: [(value + oldValue) // 2, 0]
                    })
                else:
                    self.polyPoints.get(key)[1] += 1

            lanes = combined.copy()
            cv.fillPoly(
                lanes,
                np.array([[[key, value] for key, (value, _) in self.polyPoints.items()]], dtype=np.int32),
                (0, 0, 255)
            )

        return cv.addWeighted(img, 1, lanes, 0.5, 1)

    @staticmethod
    def _segmentImage(img):
        """
        This method segments the image by extracting the ROI for lane detection.

        Args:
            img (np.ndarray): The image to extract the ROI of

        Returns:
            np.ndarray: An image only containing the pixels of the ROI
        """

        mask = np.zeros_like(img, dtype=np.uint8)

        shape = img.shape
        if len(shape) > 2:
            channel_count = shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv.fillPoly(mask, np.array([roi], dtype=np.int32), ignore_mask_color)
        return cv.bitwise_and(img, mask)

    @staticmethod
    def _filterLineShape(img):
        """
        This method uses canny to filter edges out of the image.

        Args:
            img (np.ndarray): The image to filter the edges of

        Returns:
            np.ndarray: The filtered image
        """

        gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(gray_image, (5, 5), 0)
        canny = cv.Canny(blur, 50, 150)
        return canny

    @staticmethod
    def _filterColor(img):
        """
        This method filters yellow and white out of the given image.

        Args:
            img (np.ndarray): The image to filter the color of

        Returns:
            np.ndarray: The image containing only white and yellow
        """

        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        min_yellow = np.array([20, 100, 100], dtype="uint8")
        max_yellow = np.array([100, 255, 255], dtype="uint8")

        sensitivity = 25
        min_white = np.array([0, 0, 255 - sensitivity], dtype="uint8")
        max_white = np.array([255, sensitivity, 255], dtype="uint8")

        mask_yellow = cv.inRange(img_hsv, min_yellow, max_yellow)
        mask_white = cv.inRange(img_hsv, min_white, max_white)

        mask_wy = cv.bitwise_or(mask_white, mask_yellow)

        # Use closing to close holes
        # return cv.morphologyEx(mask_wy, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=5)
        return mask_wy

    # ---------- [Object Detection] ---------- #

    def detectObjects(self, image):
        """
        This method detects other objects (cars, etc.) in the given image.
        Args:
            image (np.ndarray): The image to analyse and detect objects in

        Returns:
            np.ndarray: The image with the detected objects
        """

        # TODO: Implement
        # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
        # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
        return cv.rectangle(image, (225, 225), (275, 275), (255, 0, 0), 1)
