import cv2 as cv
import numpy as np
from typing import List, Dict
from config import *
from util import LinePoint


class Detector:
    def __init__(self):
        # This is a list, which contains a dict for each line (right/left)
        # The dict itself contains points for the x and y coordinates of the points
        # Default values are the ROI edges
        self.currentLinePoints: List[Dict[int, LinePoint]] = [
            # Right line
            {
                int(ROI[0][1]): LinePoint(ROI[0][0]),
                int(ROI[1][1]): LinePoint(ROI[1][0])
            },
            # Left line
            {
                int(ROI[2][1]): LinePoint(ROI[2][0]),
                int(ROI[3][1]): LinePoint(ROI[3][0])
            }
        ]

    # ---------- [Lane Detection] ---------- #

    def detectLines(self, img):
        """
        Detects the lines in the given image. This task is split into subtasks.

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
        lineShapes = self._filterLineShape(segmented)

        # Combine the pictures
        combined = cv.bitwise_or(color, lineShapes)

        # Perform Hough Line detection and return the overlay the method creates
        return self._houghDetection(combined)

    @staticmethod
    def _segmentImage(img):
        """
        Segments the image by extracting the ROI for lane detection.

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

        cv.fillPoly(mask, np.array([ROI], dtype=np.int32), ignore_mask_color)
        return cv.bitwise_and(img, mask)

    @staticmethod
    def _filterLineShape(img):
        """
        Uses canny to filter edges out of the image.

        Args:
            img (np.ndarray): The image to filter the edges of

        Returns:
            np.ndarray: The filtered image
        """

        # Perform canny
        canny = cv.Canny(cv.GaussianBlur(img, (5, 5), 0), 50, 150)

        # Draw a black polygon around the ROI to remove edges of the ROI
        cv.polylines(
            canny,
            np.array([ROI], dtype=np.int32),
            False,
            (0, 0, 0),
            thickness=3
        )

        return canny

    @staticmethod
    def _filterColor(img):
        """
        Filters yellow and white out of the given image.

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
        combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)),
                                   iterations=2)
        # Remove singular artifacts
        combined = cv.morphologyEx(combined, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)),
                                   iterations=1)

        return combined

    def _houghDetection(self, img):
        """
        Performs Hough Line detection on the given image and returns a line overlay which can be added onto the final
        image.

        Args:
            img (np.ndarray): The prefiltered, segmented and grayscaled image.

        Returns:
            np.ndarray: The lane overlay.
        """

        # Create the overlay image
        overlay = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        # Fill a specified polygon right in front of the car black to ignore horizontal lines
        # -> Improves performance for hough
        cv.fillPoly(
            img,
            np.array([IGNORED_ROI], dtype=np.int32),
            (0, 0, 0)
        )

        # Check if less than a fiftieth of the image is white
        if np.sum(img > 0) < WIDTH * HEIGHT // 50:
            # Use the performance variant of HoughLines
            lines = cv.HoughLinesP(img, 1, np.pi / 180, 75, maxLineGap=40)

            if lines is not None:
                # Go over every detected hough line
                for line in lines:
                    line = line[0]

                    # Get the coordinates
                    x1, y1 = (line[0], line[1])
                    x2, y2 = (line[2], line[3])

                    # Calculate the angle of the line to the image
                    lineAngle = cv.fastAtan2(y2 - y1, x2 - x1)

                    # Only vertical lines are allowed.
                    # They have an angle between 25 and 335 degrees.
                    # Other lines are considered as horizontal.
                    if 25 < lineAngle < 335:
                        # Draw detected line, if specified
                        if DRAW_HOUGH:
                            cv.line(
                                img,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                thickness=10
                            )

                        # Point is from the left line, if the angle is bigger than 270 degrees
                        isLeftLine = int(lineAngle > 270)

                        # Compare current point to existing point, if exists
                        # Do this for every line (left and right)
                        for i in range(2):
                            y = int(line[i * 2 + 1])
                            newX = int(line[i * 2])

                            # Get the current point for that y-coordinate
                            currentPoint = self.currentLinePoints[isLeftLine].get(y)

                            # Save the new value, if there is no current point, or ...
                            saveNewValue = currentPoint is None

                            if currentPoint is not None:
                                currentX = currentPoint.getX()
                                currentLifetime = currentPoint.getLifetime()

                                # ... if the euclidean distance is smaller than the tolerance, or the previous point
                                # exceeded the lifetime
                                saveNewValue = abs(newX - currentX) < LINE_TOLERANCE or currentLifetime >= MAX_LIFETIME

                                # newX = (currentX + 2 * newX) // 3

                            if saveNewValue:
                                # Save the value
                                self.currentLinePoints[isLeftLine].update({
                                    y: LinePoint(newX)
                                })

        # Create lists for storing values
        invalidPoints = []
        polyLines = []

        # Go over each line
        for index, line in enumerate(self.currentLinePoints):
            polyPoints = []

            # Go over every point in the line
            for y, linePoint in line.items():
                linePoint.increaseLifetime()

                if linePoint.getLifetime() >= MAX_LIFETIME:
                    # Schedule the point for deletion
                    invalidPoints.append(y)
                else:
                    # Schedule the point for the poly fit
                    polyPoints.append([y, linePoint.getX()])

            if len(polyPoints) > 0:
                # Estimate the polynomial function
                fittedPoly = np.polyfit(
                    [y for (y, _) in polyPoints],
                    [x for (_, x) in polyPoints],
                    2
                )
                estimator = np.poly1d(fittedPoly)

                # Get all points via the estimator
                polyLine = np.int32([
                    estimator(Y_RANGE), Y_RANGE
                ]).T

                if index == 0:
                    polyLines.extend(polyLine)
                else:
                    # Flip to ensure, that the filled poly is solid and not crossed
                    polyLines.extend(np.flipud(polyLine))

                # Draw the line marking
                cv.polylines(
                    overlay,
                    [polyLine],
                    False,
                    color=LANE_COLOR,
                    thickness=5
                )

        # Remove invalid points from the dict
        for invalidX in invalidPoints:
            for i in range(2):
                try:
                    self.currentLinePoints[i].pop(invalidX)
                except KeyError as e:
                    pass

        if len(polyLines) > 0:
            # Fill the area between the lines
            cv.fillPoly(
                overlay,
                np.array([polyLines], dtype=np.int32),
                LANE_COLOR
            )

        return overlay

    # ---------- [Object Detection] ---------- #

    def detectObjects(self, image):
        """
        Detects other objects (cars, etc.) in the given image.

        Args:
            image (np.ndarray): The image to analyse and detect objects in

        Returns:
            np.ndarray: The image with the detected objects
        """

        # TODO: Implement
        # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
        # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
        return cv.rectangle(image, (225, 225), (275, 275), (255, 0, 0), 1)
