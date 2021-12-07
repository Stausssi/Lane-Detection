import cv2 as cv


class LinePoint:
    def __init__(self, x):
        self._x = x
        self._lifetime = 0

    def getLifetime(self):
        return self._lifetime

    def increaseLifetime(self):
        self._lifetime += 1

    def getX(self):
        return self._x


def displayTextOnImage(img, text, position, font=cv.FONT_HERSHEY_PLAIN, font_scale=1.25, font_color=(0, 0, 0), thickness=2):
    cv.putText(
        img,
        text,
        position,
        font,
        font_scale,
        font_color,
        thickness
    )
