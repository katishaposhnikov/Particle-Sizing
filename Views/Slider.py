import PySimpleGUI as sg


class Slider:
    def __init__(self, name: str, key: str):
        self.slider = sg.Slider(range=(0, 255), default_value=127, orientation='horizontal', key=key,
                                enable_events=True)
        self.label = sg.Text(name)

    def layout(self):
        return [self.label, self.slider]
