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
