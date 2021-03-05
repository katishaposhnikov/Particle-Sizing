from rx.subject import Subject


class AbstractStore:
    def __init__(self, init_state, name):
        self._state = init_state
        self.name = name
        self.subject = Subject()

    @property
    def state(self):
        return self._state

    def receive_action(self, action, payload):
        pass

    def subscribe(self, callback):
        self.subject.subscribe_(on_next=callback)
