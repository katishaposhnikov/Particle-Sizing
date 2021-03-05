import constants
from models.Particle import Particle
from stores.AbstractStore import AbstractStore


class ParticleStore(AbstractStore):
    def __init__(self):
        super(ParticleStore, self).__init__(Particle(), 'particle')

    def receive_action(self, action, payload):
        if constants.IMAGE_KEY == action:
           