import cv2 as cv


class Detector:
    @staticmethod
    def detectLines(image):
        """
        This method tries to detect the lines in the given image.

        Args:
            image (cv.Mat): The given image

        Returns:
            The image with the lines drawn inside
        """

        # TODO: Implement
        # Segment the image
        image = cv.rectangle(image, (200, 200), (300, 300), (0, 0, 255), 1)
        return image

    @staticmethod
    def detectObjects(image):
        """
        This method detects other objects (cars, etc.) in the given image.

        Args:
            image: The image to analyse and detect objects in

        Returns:
            The image with the detected objects
        """

        # TODO: Implement
        # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
        # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
        return cv.rectangle(image, (225, 225), (275, 275), (255, 0, 0), 1)