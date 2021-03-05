class Settings:
    def __init__(self):
        # the number of real life millimeters that corresponds to the scale length
        self.scale_size_mm = None

        # the imaging threshold with which to process the scale. should be an int 0-255
        self.scale_threshold

        # the imgaing threshold with which to process the particle. should be an int 0-255
        self.particle_threshold

        # the radio option for how to display the processed particle
        self.masking_option
