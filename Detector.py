import cv2 as cv
import numpy as np
from typing import List, Tuple


class Detector:
    def __init__(self):
        self.prevLines = ((), ())

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

        # First, segment the image
        if segment:
            img = self._segmentImage(img)

        # Use filtering (by shape and color)
        lineShapes = self._filterLineShape(img)
        color = self._filterColor(img)

        combined = cv.addWeighted(lineShapes, 0.5, color, 1, 0)
        houghImg = combined.copy()
        combined = cv.cvtColor(combined, cv.COLOR_GRAY2RGB)

        # Regular Hough
        # lines = cv.HoughLines(houghImg, 1, np.pi/180, 150)
        # for line in lines:
        #     for r, theta in line:
        #         # Stores the value of cos(theta) in a
        #         a = np.cos(theta)
        #
        #         # Stores the value of sin(theta) in b
        #         b = np.sin(theta)
        #
        #         # x0 stores the value rcos(theta)
        #         x0 = a * r
        #
        #         # y0 stores the value rsin(theta)
        #         y0 = b * r
        #
        #         # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        #         x1 = int(x0 + 100 * (-b))
        #
        #         # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        #         y1 = int(y0 + 100 * a)
        #
        #         # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        #         x2 = int(x0 - 100 * (-b))
        #
        #         # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        #         y2 = int(y0 - 100 * a)
        #
        #         # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        #         # (0,0,255) denotes the colour of the line to be
        #         # drawn. In this case, it is red.
        #         startX = min(x1, x2)
        #         cv.line(combined, (x1, y1), (x2, y2), (0, 0, 255), 5)
        #         if startX > 100:
        #             if prevX == 0 or abs(prevX - startX) > 500:
        #                 prevX = startX

        # P variant
        lines = cv.HoughLinesP(houghImg, 1, np.pi / 180, 150)

        if lines is not None:
            drawnLines: List[Tuple[int, int]] = []
            for line in lines:
                line = line[0]
                x1, y1 = (line[0], line[1])
                x2, y2 = (line[2], line[3])

                startX = min(x1, x2)
                if abs(y1 - y2) > 50 and y1 > img.shape[0] - 250 and 100 < startX < img.shape[1] - 100:
                    # Draw line only if the startX doesnt intersect with any lines (+ thresh)
                    if not any([line1 - 100 < startX < line2 + 100 for line1, line2 in drawnLines]):
                        drawnLines.append((startX, max(x1, x2)))
                        cv.line(combined, (x1, y1), (x2, y2), (0, 0, 255), 5)

        return combined

    @staticmethod
    def _segmentImage(img):
        """
        This method segments the image by extracting the ROI for lane detection.

        Args:
            img (np.ndarray): The image to extract the ROI of

        Returns:
            np.ndarray: An image only containing the pixels of the ROI
        """

        shape = img.shape
        polyPoints = np.array(
            [[(.55 * shape[1], 0.6 * shape[0]), (shape[1], shape[0]), (0, shape[0]), (.45 * shape[1], 0.6 * shape[0])]],
            dtype=np.int32
        )

        mask = np.zeros_like(img, dtype=np.uint8)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv.fillPoly(mask, polyPoints, ignore_mask_color)
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

        sensitivity = 15
        min_white = np.array([0, 0, 255 - sensitivity], dtype="uint8")
        max_white = np.array([255, sensitivity, 255], dtype="uint8")

        mask_yellow = cv.inRange(img_hsv, min_yellow, max_yellow)
        mask_white = cv.inRange(img_hsv, min_white, max_white)

        mask_wy = cv.bitwise_or(mask_white, mask_yellow)

        # Use closing to close holes
        return cv.morphologyEx(mask_wy, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=5)

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
