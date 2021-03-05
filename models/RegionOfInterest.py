import numpy as np
from rx.subject import Subject


class RegionOfInterest:
    def __init__(self, imgbytes: np.ndarray = None, y_bounds=None, x_bounds=None):
        self.subject = Subject()

        # the raw image bytes returned by cv2.imencode('.png', resize)[1].tobytes()
        # ready to be accepted by graph.draw_image
        self.imgbytes: np.ndarray = imgbytes

        # the 'row' or 'y' bounds from the numpy array that is the original image
        self.y_bounds = y_bounds

        # the 'column' or 'x' bounds from the numpy array that is the original image
        self.x_bounds = x_bounds

        # together cv2.imread('image.png')[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]
        # gives the region of interest

        # the number of pixels that the scale occupies in the image
        self.scale_size_pixels = None

    def top_left(self):
        return self.x_bounds[0], self.y_bounds[0]

    def
