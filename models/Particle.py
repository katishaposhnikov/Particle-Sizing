import cv2
import numpy as np


class Particle:
    def __init__(self, img: np.ndarray):
        self.img: np.ndarray = img
        self.num_particle_pixels = None

    def load_image(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR)
   