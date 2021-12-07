from typing import List, Dict, Optional, Any

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

        self.currentPolynoms: List[Optional[Any]] = [None, None]

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
        img = self._segmentImage(img)

        if SHOW_HIST:
            cv.imshow("Hist", self.__getHist(img))

        # Then. filter by color
        color = self._filterColor(img)

        # And finally filter by edges (Canny)
        edges = self._filterEdges(img)

        # Combine the pictures
        combined = cv.bitwise_or(color, edges)

        # Fill a specified polygon right in front of the car black to ignore horizontal lines
        # -> Improves performance for hough
        cv.fillPoly(
            combined,
            np.array([IGNORED_ROI], dtype=np.int32),
            (0, 0, 0)
        )

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

        # TODO: maybe adjust ROI depending on curvature
        shape = img.shape
        if len(shape) > 2:
            channel_count = shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv.fillPoly(mask, np.array([ROI], dtype=np.int32), ignore_mask_color)
        return cv.bitwise_and(img, mask)

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
        canny = cv.Canny(cv.GaussianBlur(img, (5, 5), 0), 50, 150)

        # Draw a black polygon around the ROI to remove edges of the ROI
        cv.drawContours(
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

        mask_yellow = cv.inRange(img_hsv, MIN_YELLOW, MAX_YELLOW)
        mask_white = cv.inRange(img_hsv, MIN_WHITE, MAX_WHITE)

        combined = cv.bitwise_or(mask_white, mask_yellow)

        # Connect artefacts together -> strengthen the line
        combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, CROSS_FILTER_5_5, iterations=2)

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

                # Save the polynom
                self.currentPolynoms[index] = fittedPoly

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

        cars_overlay = np.zeros((HEIGHT, WIDTH, 3), dtype="uint8")

        cars = CARS_CASCADE.detectMultiScale(image, 1.15, 2)
        for (x, y, w, h) in cars:
            cv.rectangle(cars_overlay, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        return cars_overlay

        # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
        # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
