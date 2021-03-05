import PySimpleGUI as sg


class ScaleSizeInput:

    def __init__(self, scale_key: str, particle_key):
        self.input = sg.InputText(default_text='1.00', size=(5, 1), enable_events=True, key=scale_key)
        self.output = sg.InputText(default_text='', key=particle_key, readonly=True)

    def layout(self):
        return [sg.Text('3. If the scale is '),
                self.input,
                sg.Text('mm, then the particle is: '),
                self.output]
