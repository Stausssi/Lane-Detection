from typing import List, Dict, Optional, Any

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

        # This list is needed for the curvature calculation
        self.currentPolynoms: List[Optional[Any]] = [None, None]
        self.currentEstimators: List[Optional[np.poly1d]] = [None, None]

        self.ignoredRegion = [None, None]

    @staticmethod
    def __getHist(img):
        """
        Creates the histogramm of the given image

        Args:
            img (np.ndarray): The image to create the histogramm of

        Returns:
            None: Nothing
        """

        # Convert the image to HSV
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        color = {
            "b": ("Hue", [1, 256]),
            "g": ("Sat", [1, 256]),
            "r": ("Val", [1, 256])
        }

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Farbwerthistogramm")
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

        # First, segment the image
        # img = self._segmentImage(img)

        if SHOW_SEGMENTED:
            cv.imshow("Segmented", img)

        if SHOW_HIST:
            cv.imshow("Hist", self.__getHist(img))

        # Then, filter by color
        color = self._filterColor(img)

        # And finally filter by edges (Canny)
        edges = self._filterEdges(img)

        # Combine the pictures
        combined = cv.bitwise_or(color, edges)

        if SHOW_COMBINED:
            cv.imshow("Combined", combined)

        # Perform Hough Line detection and return the overlay the method creates
        return self._houghDetection(combined)

    def getCurvature(self):
        """
        Calculates the curvature of the line in the real word with the polynoms.

        Returns:
            float: The curvature, or None if no lines were detected
        """

        meterPerPixelY = 30 / HEIGHT

        # The height where the evaluation of the polynom should take place
        evaluationY = HEIGHT - CAR_HOOD_HEIGHT

        radii = []
        for fit in self.currentPolynoms:
            radii.append(
                ((1 + (2 * fit[0] * evaluationY * meterPerPixelY + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
            )

        if len(radii) > 0:
            return np.mean(radii)
        else:
            return None

    def getCarPosition(self):
        """
        Calculates the position of the car in the lane.

        Returns:
            float: The offset of the car
        """

        location = WIDTH // 2
        metersPerPixelX = 4 / WIDTH

        left_poly = self.currentPolynoms[1]
        right_poly = self.currentPolynoms[0]

        bottom_left = left_poly[0] * HEIGHT ** 2 + left_poly[1] * HEIGHT + left_poly[2]
        bottom_right = right_poly[0] * HEIGHT ** 2 + right_poly[1] * HEIGHT + right_poly[2]

        lane_center = (bottom_right - bottom_left) / 2 + bottom_left

        return round((location - lane_center) * metersPerPixelX, 2)

    # def _segmentImage(self, img):
    #     """
    #     Segments the image by extracting the ROI for lane detection.
    #
    #     Args:
    #         img (np.ndarray): The image to extract the ROI of
    #
    #     Returns:
    #         np.ndarray: An image only containing the pixels of the ROI
    #     """
    #
    #     mask = np.zeros_like(img, dtype=np.uint8)
    #
    #     shape = img.shape
    #     if len(shape) > 2:
    #         channel_count = shape[2]
    #         ignore_mask_color = (255,) * channel_count
    #     else:
    #         ignore_mask_color = 255
    #
    #     cv.fillPoly(mask, np.array([self.adjustedROI], dtype=np.int32), ignore_mask_color)
    #     return cv.bitwise_and(img, mask)

    @staticmethod
    def _filterEdges(img):
        """
        Uses canny to filter edges out of the image.

        Args:
            img (np.ndarray): The image to filter the edges of

        Returns:
            np.ndarray: The filtered image
        """

        # Perform canny
        return cv.Canny(cv.GaussianBlur(img, (5, 5), 0), 50, 150)

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

        mask_yellow = cv.inRange(img_hsv, MIN_YELLOW, MAX_YELLOW)
        mask_white = cv.inRange(img_hsv, MIN_WHITE, MAX_WHITE)

        combined = cv.bitwise_or(mask_white, mask_yellow)

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

        numUpdatedPoints = 0

        # Check if less than a third of the image is white
        if np.sum(img > 0) < WIDTH * HEIGHT // 3:
            # Use the performance variant of HoughLines
            lines = cv.HoughLinesP(img, 1, np.pi / 180, 150, maxLineGap=75, minLineLength=50)

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
                    # They have an angle between 15 and 345 degrees.
                    # Other lines are considered as horizontal.
                    if 15 < lineAngle < 345:
                        # Draw detected line, if specified
                        if DRAW_HOUGH:
                            cv.line(
                                overlay,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                thickness=10
                            )

                        # Point is from the left line, if the angle is bigger than 270 degrees
                        isLeftLine = int(any([x < WIDTH // 2 for x in [x1, x2]]))
                        # TODO: Sliding window ?

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

                                xDistance = abs(newX - currentX)

                                # ... if the euclidean distance is bigger than the tolerance, or the previous point
                                # exceeded the lifetime
                                saveNewValue = xDistance > LINE_TOLERANCE or currentLifetime >= MAX_LIFETIME
                                if xDistance < LINE_TOLERANCE:
                                    currentPoint.decreaseLifetime()

                                # newX = (currentX + 2 * newX) // 3

                            if saveNewValue:
                                numUpdatedPoints += 1
                                # Save the value
                                self.currentLinePoints[isLeftLine].update({
                                    y: LinePoint(newX)
                                })
        else:
            print("too much white")

        totalPoints = len(self.currentLinePoints[0]) + len(self.currentLinePoints[1])

        # Create lists for storing values
        invalidPoints = []
        polyLines = []

        # Go over each line
        for index, line in enumerate(self.currentLinePoints):
            polyPointsY = []
            polyPointsX = []

            # Go over every point in the line
            for y, linePoint in line.items():
                linePoint.increaseLifetime()

                if linePoint.getLifetime() >= MAX_LIFETIME:
                    # Schedule the point for deletion
                    invalidPoints.append(y)
                else:
                    # Schedule the point for the poly fit
                    polyPointsY.append(y)
                    polyPointsX.append(linePoint.getX())

            if len(polyPointsY) > 0:
                if numUpdatedPoints > totalPoints / 10 or self.currentEstimators[index] is None:
                    # Estimate the polynomial function if many points changed
                    fittedPoly = np.polyfit(
                        polyPointsY,
                        polyPointsX,
                        2
                    )
                    estimator = np.poly1d(fittedPoly)

                    # Save the polynom
                    self.currentPolynoms[index] = fittedPoly
                    self.currentEstimators[index] = estimator
                else:
                    estimator = self.currentEstimators[index]

                # Get all points via the estimator
                polyLine = np.int32([
                    estimator(Y_RANGE), Y_RANGE
                ]).T

                if index == 0:
                    polyLines.extend(polyLine)
                else:
                    # Flip to ensure, that the filled poly is solid and not crossed
                    polyLines.extend(np.flipud(polyLine))

        # Remove invalid points from the dict
        for invalidX in invalidPoints:
            for i in range(2):
                try:
                    self.currentLinePoints[i].pop(invalidX)
                except KeyError:
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

    @staticmethod
    def detectObjects(image):
        """
        Detects objects (signs, cars, etc.) in the given image.

        Args:
            image (np.ndarray): The image to perform the detection on.

        Returns:
            np.ndarray: An overlay with boxes for every detected object
        """

        car_overlay = Detector.detectCars(image)
        sign_overlay = Detector.detectSigns(image)

        return cv.bitwise_or(car_overlay, sign_overlay)

    @staticmethod
    def detectCars(image):
        """
        Detects cars in the given image.

        Args:
            image (np.ndarray): The image to analyse and detect cars in

        Returns:
            np.ndarray: An overlay with the detected cars
        """

        cars_overlay = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        # cars = CARS_CASCADE.detectMultiScale(image, 1.15, 2)
        # for (x, y, w, h) in cars:
        #     cv.rectangle(cars_overlay, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        return cars_overlay

        # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
        # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/

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
