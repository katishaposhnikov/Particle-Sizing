import PySimpleGUI as sg
import cv2


class Application:
    def __init__(self):

        sg.theme('BluePurple')

        self.graph_size = (400, 400)

        layout = self.get_layout()

        window = sg.Window('Particle Sizing', layout)
        orig_graph = window['-ORIGINAL-']
        clean_graph = window['-CLEANED-']

        orig_image = None
        dragging = False
        start_point = end_point = prior_rect = prior_clean_img = None

        while True:  # Event Loop
            event, values = window.read()
            # print(event, values)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == '-FILENAME-':
                orig_image = cv2.imread(values['-FILENAME-'], cv2.IMREAD_COLOR)
                img_bytes, location = self.scale_img_to_graph(orig_image, orig_graph)
                orig_graph.draw_image(data=img_bytes, location=location)
            elif event == '-DONE-':
                # get region of interest
                roi = self.get_region_of_interest(orig_graph, prior_rect, orig_image)
                roi_bytes, location = self.scale_img_to_graph(roi, clean_graph)
                if prior_clean_img:
                    clean_graph.delete_figure(prior_clean_img)
                prior_clean_img = clean_graph.draw_image(data=roi_bytes, location=location)
                # Blur image
                # blur = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)[1]
                # blur = cv2.GaussianBlur(blur,(5,5),0)
                # get connected components
                # num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(blur, connectivity=4)
                # find the longest/widest one (assume that one is scale)
                # lengths_and_width
                # index = np.argmax(stats[])
                # highlight it bright green on the left graph
                # calculate real-life size of pixel
                # remove highlighted area
                # apply threshold
                # calculate area
                # reflect changes in gui
            elif event == '-ORIGINAL-':
                x, y = values["-ORIGINAL-"]
                if not dragging:
                    start_point = (x, y)
                    dragging = True
                else:
                    end_point = (x, y)
                if prior_rect:
                    orig_graph.delete_figure(prior_rect)
                if None not in (start_point, end_point):
                    prior_rect = orig_graph.draw_rectangle(start_point, end_point, line_color='red')
            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False

        window.close()

    def get_scale_data(self, img, graph):
        old_height, old_width = img.shape[0], img.shape[1]
        graph_width, graph_height = graph.get_size()
        height_ratio = graph_height / old_height
        width_ratio = graph_width / old_width
        scale = min(height_ratio, width_ratio)

        new_height = int(old_height * scale)
        new_width = int(old_width * scale)

        x_offset = int((graph_width - new_width) / 2)
        y_offset = int((graph_height - new_height) / 2)

        return (scale, x_offset, y_offset)

    def scale_img_to_graph(self, img, graph):
        scale, x_offset, y_offset = self.get_scale_data(img, graph)
        old_height, old_width = img.shape[0], img.shape[1]

        new_height = int(old_height * scale)
        new_width = int(old_width * scale)

        resize = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img_bytes = cv2.imencode('.png', resize)[1].tobytes()

        location = (x_offset, graph.get_size()[0] - y_offset)

        return img_bytes, location

    def img_coords_to_graph_coords(self, xy, img, graph):
        scale, x_offset, y_offset = self.get_scale_data(img, graph)
        nx = xy[0] * scale + x_offset
        ny = xy[1] * scale + y_offset
        return nx, ny

    def graph_coords_to_image_coords(self, xy, img, graph):
        scale, x_offset, y_offset = self.get_scale_data(img, graph)
        nx = (xy[0] - x_offset) / scale
        ny = (xy[1] - y_offset) / scale
        return nx, ny

    def get_region_of_interest(self, graph, rect_id, img):
        upper_left, lower_right = graph.get_bounding_box(rect_id)

        upper_left_image = self.graph_coords_to_image_coords(upper_left, img=img, graph=graph)
        lower_right_image = self.graph_coords_to_image_coords(lower_right, img=img, graph=graph)

        y1 = int(img.shape[0] - upper_left_image[1])
        y2 = int(img.shape[0] - lower_right_image[1])
        x1 = int(upper_left_image[0])
        x2 = int(lower_right_image[0])

        return img[y1:y2, x1:x2]

    def get_layout(self):
        return [[sg.Text('1. Please select a photo:'),
                 sg.InputText(size=(50, 1), key='-FILENAME-', enable_events=True, readonly=True),
                 sg.FileBrowse()],
                [sg.Text('2. Please highlight the text and scale with your mouse.'), sg.Button('Done', key='-DONE-')],
                [self.create_original_image(), sg.VSep(), self.create_cleaned_image()],
                [sg.Text('5. Particle size is: ', key='-SIZE')],
                [sg.Text('6. Choose another image.')]
                ]

    def create_original_image(self):
        return sg.Graph(key='-ORIGINAL-', canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size,
                        border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)

    def create_cleaned_image(self):
        return sg.Graph(key='-CLEANED-', canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size,
                        border_width=1)


# start the app
print("[INFO] starting...")
pba = Application()
