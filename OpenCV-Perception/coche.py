class coche:
    _counter = 0
    _list_car = []
    def __init__(self, center, area):
        _counter += 1
        self.id = id._counter
        self.center = center
        self.area = area
        _list_car.append(self)