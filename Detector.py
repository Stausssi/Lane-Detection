import matplotlib.pylab as plt
import cv2 as cv
import numpy as np


class Detector:
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
        segmented = self.__segmentImage(img)

        # Use filtering (by shape and color)
        lineShapes = self.__filterLineShape(segmented)
        color = self.__filterColor(segmented)

        return cv.addWeighted(lineShapes, 0.5, color, 1, 0)

    @staticmethod
    def __segmentImage(img):
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
    def __filterLineShape(img):
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
    def __filterColor(img):
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
