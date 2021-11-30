import cv2 as cv
import numpy as np
from typing import List, Dict
from config import *
from util import LinePoint


class Detector:
    def __init__(self):
        # This is a list, which contains a dict for each line (right/left)
        # The dict itself contains points for the x and y coordinates of the points
        self.currentLinePoints = [
            # Right line
            {
                int(roi[0][1]): LinePoint(roi[0][0]),
                int(roi[1][1]): LinePoint(roi[1][0])
            },
            # Left line
            {
                int(roi[2][1]): LinePoint(roi[2][0]),
                int(roi[3][1]): LinePoint(roi[3][0])
            }
        ]

    # ---------- [Lane Detection] ---------- #

    def detectLines(self, img):
        """
        This method detects the lines in the given image. This task is split into subtasks.

        Args:
            img (np.ndarray): The image to detect lines on

        Returns:
            np.ndarray: The image with lines drawn onto it
        """

        # First, segment the image
        segmented = self._segmentImage(img)

        # Then filter by color
        color = self._filterColor(segmented)

        # Filter by shapes (Canny)
        lineShapes = self._segmentImage(self._filterLineShape(img))

        # Combine the pictures
        combined = cv.addWeighted(lineShapes, 1, color, 1, 0)

        # Performance variant
        lines = cv.HoughLinesP(combined, 1, np.pi / 180, 150, maxLineGap=25)

        if lines is not None:
            # Go over every hough line
            for line in lines:
                line = line[0]
                x1, y1 = (line[0], line[1])
                x2, y2 = (line[2], line[3])

                if abs(y1 - y2) > 50:
                    # Calculate the angle of the line to the image
                    lineAngle = cv.fastAtan2(y2 - y1, x2 - x1)

                    isLeftLine = int(lineAngle > 270)

                    # Compare current point to existing point, if exists
                    for i in range(2):
                        y = int(line[i * 2 + 1])
                        newX = int(line[i * 2])
                        currentPoint = self.currentLinePoints[isLeftLine].get(i * 2 + 1)

                        saveNewValue = currentPoint is None

                        if currentPoint is not None:
                            currentX = currentPoint.getX()

                            saveNewValue = abs(newX - currentX) < lineTolerance or currentPoint.getLifetime() >= maximumLifetime

                        if saveNewValue:
                            self.currentLinePoints[isLeftLine].update({
                                y: LinePoint(newX)
                            })

        invalidPoints = []
        polyLines = []
        for index, line in enumerate(self.currentLinePoints):
            polyPoints = []
            for y, linePoint in line.items():
                linePoint.increaseLifetime()

                if linePoint.getLifetime() >= maximumLifetime:
                    invalidPoints.append(y)
                else:
                    polyPoints.append([y, linePoint.getX()])

            if len(polyPoints) > 0:
                # Estimate the polynomial function
                fittedPoly = np.polyfit(
                    [y for (y, _) in polyPoints],
                    [x for (_, x) in polyPoints],
                    2
                )
                estimator = np.poly1d(fittedPoly)

                # Get all points
                polyLine = np.int32([
                    estimator(range_y), range_y
                ]).T

                if index == 0:
                    polyLines.extend(polyLine)
                else:
                    # Flip to ensure, that the filled poly is solid and not crossed
                    polyLines.extend(np.flipud(polyLine))

                cv.polylines(
                    img,
                    [polyLine],
                    False,
                    color=(0, 255, 255),
                    thickness=5
                )

        for invalidX in invalidPoints:
            for i in range(2):
                try:
                    self.currentLinePoints[i].pop(invalidX)
                except KeyError as e:
                    pass

        lane_img = img.copy()

        if len(self.currentLinePoints[0]) > 0 and len(self.currentLinePoints[1]) > 0:
            # TODO: Fix buggy drawing -> sort ?
            cv.fillPoly(
                lane_img,
                np.array([polyLines], dtype=np.int32),
                (0, 0, 255)
            )

        return cv.addWeighted(img, 1, lane_img, 0.5, 1)

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

        # gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(img, (5, 5), 0)
        return cv.Canny(blur, 50, 150)

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

        combined = cv.bitwise_or(mask_white, mask_yellow)

        # Connect artefacts together -> strengthen the line
        combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=2)
        # Remove singular artifacts
        combined = cv.morphologyEx(combined, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=1)

        return combined

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
