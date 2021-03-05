from models.Settings import Settings
from stores.AbstractStore import AbstractStore


class SettingsStore(AbstractStore):
    def __init__(self, ):
        super(SettingsStore, self).__init__(Settings(), 'settings')

    def receive_action(self, action, payload):
        pass
