class LinePoint:
    def __init__(self, y):
        self._y = y
        self._lifetime = 0

    def getLifetime(self):
        return self._lifetime

    def increaseLifetime(self):
        self._lifetime += 1

    def getY(self):
        return self._y
