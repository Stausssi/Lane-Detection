from typing import List, Dict, Optional, Any, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from config import *
from util import LinePoint


class Detector:
    def __init__(self):
        # This is a list, which contains a dict for each line (right/left)
        # The dict itself contains points for the x and y coordinates of the points
        # Default values are the ROI edges
        self.currentLinePoints: List[Dict[int, LinePoint]] = [
            # Right line
            {},
            # Left line
            {}
        ]

        # This list is needed for the curvature calculation
        self.currentPolynomials: List[Optional[Any]] = [None, None]
        self.currentPolyLines: List[Optional[List[Tuple[int, int]]]] = [None, None]

    @staticmethod
    def __getHist(img):
        """
        Creates the histogram of the given image

        Args:
            img (np.ndarray): The image to create the histogram of

        Returns:
            None: Nothing
        """

        # Convert the image to HSV
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        color = {
            "b": ("Hue", [1, 256]),
            "g": ("Sat", [1, 256]),
            "r": ("Val", [1, 256])
        }

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("HSV Histogram")
        ax.set_xlim([0, 256])

        for i, (col, (label, ranges)) in enumerate(color.items()):
            hist = cv.calcHist([img], [i], None, [256], ranges)
            ax.plot(hist, color=col, label=label)

        fig.legend()
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return img

    # ---------- [Lane Detection] ---------- #

    def detectLines(self, img):
        """
        Detects the lines in the given image. This task is split into subtasks.

        Args:
            img (np.ndarray): The image to detect lines on

        Returns:
            np.ndarray: The image with lines drawn onto it
        """

        # Show histogram if wanted
        SHOW_HIST and cv.imshow("Hist", self.__getHist(img))

        # Filtering by color is more than enough for the birds eye view
        color = self._filterColor(img)

        SHOW_FILTERED and cv.imshow("Filtered", color)

        # Perform Hough Line detection and return the overlay the method creates
        return self._houghDetection(color)

    def getCurvature(self):
        """
        Calculates the curvature of the line in the real word with the polynomials.

        Returns:
            float: The curvature, or None if no lines were detected
        """

        radii = []
        for fit in self.currentPolynomials:
            radii.append(
                ((1 + (2 * fit[0] * EVALUATION_Y * MPP_Y + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
            )

        if len(radii) > 0:
            return np.mean(radii)
        else:
            return None

    def getCenterOffset(self):
        """
        Calculates the position offset of the car in the lane.

        Returns:
            float: The offset of the car
        """

        left_poly = self.currentPolynomials[1]
        right_poly = self.currentPolynomials[0]

        bottom_left = left_poly[0] * HEIGHT ** 2 + left_poly[1] * HEIGHT + left_poly[2]
        bottom_right = right_poly[0] * HEIGHT ** 2 + right_poly[1] * HEIGHT + right_poly[2]

        lane_center = (bottom_right - bottom_left) / 2 + bottom_left

        return round((DEFAULT_CENTER - lane_center) * MPP_X, 2)

    @staticmethod
    def _filterColor(img):
        """
        Filters yellow and white markings out of the given image.

        Args:
            img (np.ndarray): The image to filter the color of

        Returns:
            np.ndarray: The image containing only white and yellow
        """

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        mask_yellow = cv.inRange(img_hsv, MIN_YELLOW, MAX_YELLOW)
        mask_white = cv.inRange(img_hsv, MIN_WHITE, MAX_WHITE)

        combined = cv.bitwise_or(mask_white, mask_yellow)

        return combined

    def _houghDetection(self, img):
        """
        Performs Hough Line detection on the given image and returns a line overlay which can be added onto the final
        image.

        Args:
            img (np.ndarray): The prefiltered, segmented and gray-scaled image.

        Returns:
            np.ndarray: The lane overlay.
        """

        # Create the overlay image
        overlay = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        numUpdatedPoints = 0

        # Check if less than a third of the image is white
        if np.sum(img > 0) < WIDTH * HEIGHT // 3:
            # Use the performance variant of HoughLines
            lines = cv.HoughLinesP(img, 1, np.pi / 180, 150, minLineLength=50, maxLineGap=50)

            if lines is not None:
                # Go over every detected hough line
                for line in lines:
                    line = line[0]

                    # Get the coordinates
                    x1, y1 = (line[0], line[1])
                    x2, y2 = (line[2], line[3])

                    # Point is from the left line, if it's left from the middle
                    isLeftLine = int(any([x < DEFAULT_CENTER for x in [x1, x2]]))

                    # Compare current point to existing point, if exists
                    # Do this for every point (start and end)
                    for x, y in [(x1, y1), (x2, y2)]:
                        # Get the current point for that y-coordinate
                        currentPoint = self.currentLinePoints[isLeftLine].get(y)

                        # Save the new value, if there is no current point, or ...
                        saveNewValue = currentPoint is None

                        pointScore = 1
                        if currentPoint is not None:
                            currentX = currentPoint.getX()
                            currentLifetime = currentPoint.getLifetime()

                            xDistance = abs(x - currentX)
                            pointScore = max(x, currentX) / (min(x, currentX) + 1)

                            # ... if the euclidean distance is bigger than the tolerance, or the previous point
                            # exceeded the lifetime
                            saveNewValue = xDistance > LINE_TOLERANCE or currentLifetime > 0
                            if xDistance < LINE_TOLERANCE:
                                currentPoint.increaseLifetime()

                        if saveNewValue:
                            numUpdatedPoints += pointScore
                            # Save the value
                            self.currentLinePoints[isLeftLine].update({
                                y: LinePoint(x)
                            })
        else:
            print("too much white")

        totalPoints = len(self.currentLinePoints[0]) + len(self.currentLinePoints[1])

        # Create lists for storing the lines
        polyLines = []

        # Go over each line
        for index, line in enumerate(self.currentLinePoints):
            # Estimate a new polynomial if many points changed
            fitNewPoly = numUpdatedPoints > totalPoints / TOTAL_POINTS_DIVIDER or self.currentPolyLines[index] is None

            # Store points for the new polynomial
            polyPointsY = []
            polyPointsX = []

            # Go over every point in the line
            for y, linePoint in line.items():
                linePoint.decreaseLifetime()

                if linePoint.getLifetime() > 0 and fitNewPoly:
                    # Schedule the point for the poly fit
                    polyPointsY.append(y)
                    polyPointsX.append(linePoint.getX())

            if fitNewPoly and len(polyPointsY) > 0:
                fittedPoly = np.polyfit(
                    polyPointsY,
                    polyPointsX,
                    2
                )
                estimator = np.poly1d(fittedPoly)

                # Get all points via the estimator
                polyLine = np.int32([
                    estimator(Y_RANGE), Y_RANGE
                ]).T

                # Save the polynomial
                self.currentPolynomials[index] = fittedPoly
                self.currentPolyLines[index] = polyLine
            else:
                polyLine = self.currentPolyLines[index]

            if index == 0:
                polyLines.extend(polyLine)
            else:
                # Flip to ensure, that the filled poly is solid and not crossed
                polyLines.extend(np.flipud(polyLine))

        if len(polyLines) > 0:
            # Fill the area between the lines
            cv.fillPoly(
                overlay,
                np.array([polyLines], dtype=np.int32),
                LANE_COLOR
            )

        return overlay

    # ---------- [Sign Detection] ---------- #

    @staticmethod
    def detectSigns(image):
        """
        Detects road signs in the given image.

        Args:
            image (np.ndarray): The image to analyse and detect cars in

        Returns:
            np.ndarray: An overlay with the detected signs
        """

        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        sign_overlay = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        # Fill ROI with black to prevent lane markings to interfere
        cv.fillPoly(
            image,
            np.array([ROI], dtype=np.int32),
            (0, 0, 0)
        )
        cv.rectangle(
            image,
            (0, HEIGHT),
            (WIDTH, HEIGHT - 150),
            (0, 0, 0),
            cv.FILLED
        )

        # Green signs
        sign_image = cv.inRange(image, MIN_GREEN_SIGN, MAX_GREEN_SIGN)
        # Yellow signs
        sign_image = cv.bitwise_or(sign_image, cv.inRange(image, MIN_YELLOW_SIGN, MAX_YELLOW_SIGN))

        contours, _ = cv.findContours(sign_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) > 75:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(sign_overlay, (x, y), (x + w, y+h), (0, 255, 0), 3)
                cv.putText(
                    sign_overlay,
                    "Sign",
                    (x, y - 15),
                    cv.FONT_HERSHEY_PLAIN,
                    1.25,
                    (0, 255, 0),
                    1
                )

        return sign_overlay
