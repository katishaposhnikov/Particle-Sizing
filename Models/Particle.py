import numpy as np


class Particle:
    def __init__(self, img: np.ndarray):
        self.img: np.ndarray = img
        self.num_particle_pixels = None
