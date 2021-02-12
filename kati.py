import PySimpleGUI as sg


# import cv2


class Application:
    def __init__(self):
        sg.theme('BluePurple')

        layout = self.get_layout()

        window = sg.Window('Particle Sizing', layout)

        orig_image = window['-ORIGINAL-']
        clean_image = window['-CLEANED-']
        self.orig_filename = None
        dragging = False
        start_point = end_point = prior_rect = None

        while True:  # Event Loop
            event, values = window.read()
            print(event, values)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == '-FILENAME-':
                orig_image.draw_image(filename=values['-FILENAME-'], location=(0, 400))
            elif event == '-ORIGINAL-':
                x, y = values["-ORIGINAL-"]
                if not dragging:
                    start_point = (x, y)
                    dragging = True
                else:
                    end_point = (x, y)
                if prior_rect:
                    orig_image.delete_figure(prior_rect)
                if None not in (start_point, end_point):
                    prior_rect = orig_image.draw_rectangle(start_point, end_point, line_color='red')

            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False

        window.close()

    def get_layout(self):
        return [[sg.Text('1. Please select a photo:'),
                 sg.InputText(size=(50, 1), key='-FILENAME-', enable_events=True, readonly=True),
                 sg.FileBrowse()],
                [sg.Text('2. Please highlight the text and scale with your mouse.'), sg.Button('Done')],
                [self.create_original_image(), sg.VSep(), self.create_cleaned_image()],
                [sg.Text('5. Particle size is: ', key='-SIZE')],
                [sg.Text('6. Choose another image.')]
                ]

    def create_original_image(self):
        return sg.Graph(key='-ORIGINAL-', canvas_size=(400, 400), graph_bottom_left=(0, 0), graph_top_right=(400, 400),
                        border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)

    def create_cleaned_image(self):
        return sg.Graph(key='-CLEANED-', canvas_size=(400, 400), graph_bottom_left=(0, 0), graph_top_right=(400, 400),
                        border_width=1)


# start the app
print("[INFO] starting...")
pba = Application()
