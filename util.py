import cv2 as cv


from config import MAX_LIFETIME


class LinePoint:
    """
    Represents a point

    Args:
        x (int): The x coordinate of the point
    """
    def __init__(self, x):
        self._x = x
        self._lifetime = MAX_LIFETIME

    def getLifetime(self):
        """
        Returns the lifetime of this point.

        Returns:
            int: The lifetime
        """

        return self._lifetime

    def increaseLifetime(self):
        """
        Increases the lifetime of the point.

        Returns:
            None: Nothing
        """

        self._lifetime += 1

    def decreaseLifetime(self):
        """
        Decreases the lifetime of the point.

        Returns:
            None: Nothing
        """

        self._lifetime -= 1

    def getX(self):
        """
        Returns the x-coordinate of the point.

        Returns:
            int: The x-coordinate
        """

        return self._x


def displayTextOnImage(
        img,
        text,
        position,
        font=cv.FONT_HERSHEY_PLAIN,
        font_scale=1.25,
        font_color=(0, 0, 0),
        thickness=2):
    """
    Displays a given text on an image.

    Args:
        img (np.ndarray): The image to draw the text onto
        text (str): The text to draw
        position (tuple[int, int]): The position of the upper left corner of the text
        font: The font of the text
        font_scale (float): The scale of the text
        font_color (tuple[int, int, int]): The color of the text
        thickness (int): The thickness of the text

    Returns:
        None: Nothing
    """

    cv.putText(
        img,
        text,
        position,
        font,
        font_scale,
        font_color,
        thickness
    )
