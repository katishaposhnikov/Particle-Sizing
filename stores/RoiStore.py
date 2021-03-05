from models.RegionOfInterest import RegionOfInterest
from stores.AbstractStore import AbstractStore


class RoiStore(AbstractStore):
    def __init__(self, ):
        super(RoiStore, self).__init__(RegionOfInterest(), 'roi')

    def receive_action(self, action, payload):
        pass
