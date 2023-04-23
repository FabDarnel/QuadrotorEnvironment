# drone.py
class Drone:
    def __init__(self, id, position, orientation):
        self.id = id
        self.position = position
        self.orientation = orientation

    def set_position(self, position):
        self.position = position

    def set_orientation(self, orientation):
        self.orientation = orientation

    def get_position(self):
        return self.position

    def get_orientation(self):
        return self.orientation