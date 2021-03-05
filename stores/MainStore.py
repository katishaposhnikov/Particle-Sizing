from stores.AbstractStore import AbstractStore


class MainStore(AbstractStore):
    def __init__(self, stores):
        super(MainStore, self).__init__({store.name: store.state for store in stores}, 'main')
    
    def receive_action(self, action, payload):
        for store in self._state:
            store.receive_action(action, payload)
        self.subject.on_next(self.state)
