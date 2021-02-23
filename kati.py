import PySimpleGUI as sg
import cv2
import numpy as np


class Application:
    def __init__(self):

        sg.theme('BluePurple')

        self.graph_size = (600, 400)
        self.IMAGE_ALGO_SELECTION = '-IMAGE-ALGO-SELECTION-'
        self.SCALE_ALGO_SELECTION = '-SCALE-ALGO-SELECTION-'
        self.SCALE_ALGOS = [{'name': 'Color Threshold', 'func': self.scale_color_threshold}, {
            'name': 'Grayscale Threshold', 'func': self.scale_grayscale_thresholding}]
        self.IMAGE_ALGOS = [{'name': 'Color Threshold', 'func': self.image_color_threshold}, {
            'name': 'Grayscale Threshold', 'func': self.image_sharpen_plus_threshold}]
        self.ORIGINAL_KEY = '-ORIGINAL-'
        self.CLEANED_KEY = '-CLEANED-'
        self.PARTICLE_SIZE_KEY = '-SIZE-'
        self.FILEINPUT_KEY = '-FILENAME-'
        self.DONE_KEY = '-DONE-'

        layout = self.get_layout()

        window = sg.Window('Particle Sizing', layout)
        orig_graph = window[self.ORIGINAL_KEY]
        clean_graph = window[self.CLEANED_KEY]
        particle_size = window[self.PARTICLE_SIZE_KEY]
        scale_algo_chooser = window[self.SCALE_ALGO_SELECTION]
        image_algo_chooser = window[self.IMAGE_ALGO_SELECTION]

        orig_image_id = None
        orig_image = None
        dragging = False
        start_point = end_point = prior_rect = None

        while True:  # Event Loop
            event, values = window.read()
            # print(event, values)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == self.FILEINPUT_KEY:
                orig_image = cv2.imread(
                    values[self.FILEINPUT_KEY], cv2.IMREAD_COLOR)
                img_bytes, location = self.scale_img_to_graph(
                    orig_image, orig_graph)
                if orig_image_id:
                    orig_graph.delete_figure(orig_image_id)
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)
            elif event == self.DONE_KEY:
                # get region of interest
                if not prior_rect:
                    continue
                roi, ys, xs = self.get_region_of_interest(
                    orig_graph, prior_rect, orig_image)

                roi_processed = None
                for algo in self.SCALE_ALGOS:
                    if algo['name'] == scale_algo_chooser.get():
                        roi_processed = algo['func'](roi)
                        break

                # get connected components
                num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    roi_processed, connectivity=4)

                # get the lengths and width of the connected components (0 is background so exclude it)
                lengths_and_widths = np.stack(
                    (stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]))

                # find the longest/widest one (assume that one is the image scale)
                largest_cc = (np.argmax(lengths_and_widths) % lengths_and_widths.shape[
                    1]) + 1  # add 1 because we ignored background

                # highlight it bright red on the left graph
                # In HSV 179 is red
                label_hue = np.uint8(np.where(labels == largest_cc, 179, 0))
                blank_ch = 255 * np.ones_like(label_hue)
                labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

                # Converting cvt to BGR
                labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
                labeled_img[label_hue == 0] = 0

                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(labeled_img, labeled_img, mask=mask)

                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg, img2_fg)
                new_img = orig_image.copy()
                new_img[ys[0]:ys[1], xs[0]:xs[1]] = dst

                img_bytes, location = self.scale_img_to_graph(
                    new_img, orig_graph)
                orig_graph.erase()
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)

                # calculate real-life size of pixel
                scale_size = np.max(lengths_and_widths)
                pixel_size_in_microns = 1000 / scale_size

                # calculate number of pixels in particle
                particle = orig_image.copy()
                for algo in self.IMAGE_ALGOS:
                    if algo['name'] == image_algo_chooser.get():
                        particle = algo['func'](particle, labels)
                        break

                # black-out the user selection rectangle so that it isn't counted in particle pixel area calculation
                particle[ys[0] - 2:ys[1] + 2, xs[0] - 2:xs[1] + 2] = 0

                # count number of non-zero (non-black) pixels which should only be the pixels of the particle
                num_particle_pixels = cv2.countNonZero(particle)
                # calculate area
                particle_area = (
                    pixel_size_in_microns * pixel_size_in_microns * num_particle_pixels) / 1000000
                particle_area = round(particle_area, 4)

                # reflect changes in gui
                particle_size.update(
                    value=f'3. Particle size is: {particle_area} mm2')
                particle_bytes, location = self.scale_img_to_graph(
                    particle, clean_graph)
                clean_graph.erase()
                clean_graph.draw_image(data=particle_bytes, location=location)
            elif event == self.ORIGINAL_KEY:
                x, y = values[self.ORIGINAL_KEY]
                if not dragging:
                    start_point = (x, y)
                    dragging = True
                else:
                    end_point = (x, y)
                if prior_rect:
                    orig_graph.delete_figure(prior_rect)
                    prior_rect = None
                if None not in (start_point, end_point):
                    prior_rect = orig_graph.draw_rectangle(
                        start_point, end_point, line_color='red')
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

        return scale, x_offset, y_offset

    def scale_img_to_graph(self, img, graph):
        scale, x_offset, y_offset = self.get_scale_data(img, graph)
        old_height, old_width = img.shape[0], img.shape[1]

        new_height = int(old_height * scale)
        new_width = int(old_width * scale)

        resize = cv2.resize(img, (new_width, new_height),
                            interpolation=cv2.INTER_AREA)
        img_bytes = cv2.imencode('.png', resize)[1].tobytes()

        location = (x_offset, graph.get_size()[1] - y_offset)

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

        upper_left_image = self.graph_coords_to_image_coords(
            upper_left, img=img, graph=graph)
        lower_right_image = self.graph_coords_to_image_coords(
            lower_right, img=img, graph=graph)

        y1 = int(img.shape[0] - upper_left_image[1])
        y2 = int(img.shape[0] - lower_right_image[1])
        x1 = int(upper_left_image[0])
        x2 = int(lower_right_image[0])

        return img[y1:y2, x1:x2], (y1, y2), (x1, x2)

    def get_layout(self):
        return [[sg.Text('1. Please select a photo:'),
                 sg.InputText(size=(50, 1), key='-FILENAME-',
                              enable_events=True, readonly=True),
                 sg.FileBrowse()],
                [sg.Text('2. Please highlight the text and scale with your mouse.'),
                 sg.Button('Done', key='-DONE-')],
                [sg.Text('3. Select an algorithm for filtering the scale.'),
                 sg.Combo(values=[x['name'] for x in self.SCALE_ALGOS], default_value=self.SCALE_ALGOS[0]['name'],
                          key=self.SCALE_ALGO_SELECTION, readonly=True)],
                [sg.Text('4. Select an algorithm for filtering the image.'),
                 sg.Combo(values=[x['name'] for x in self.IMAGE_ALGOS], default_value=self.IMAGE_ALGOS[0]['name'],
                          key=self.IMAGE_ALGO_SELECTION, readonly=True)],
                [self.create_original_image(), sg.VSep(), self.create_cleaned_image()],
                [sg.InputText(default_text='5. Particle size is: ',
                              key='-SIZE-', readonly=True)],
                [sg.Text('6. Choose another image.')]
                ]

    def create_original_image(self):
        return sg.Graph(key=self.ORIGINAL_KEY, canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size,
                        border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)

    def create_cleaned_image(self):
        return sg.Graph(key='-CLEANED-', canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size, background_color='lightblue',
                        border_width=1)

    def scale_grayscale_thresholding(self, roi):
        # Blur image
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        return cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def scale_color_threshold(self, roi):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        return cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def image_sharpen_plus_threshold(self, particle, labels):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        particle = cv2.filter2D(particle, -1, kernel)
        particle = cv2.cvtColor(particle, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(particle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def image_color_threshold(self, particle, labels):
        target = cv2.cvtColor(particle, cv2.COLOR_BGR2HSV)
        # find a good hsv range for the background color
        # filter it out

        # or find a good hsv range for the particle
        # filter it in and everything else out
        return roi


# start the app
print("[INFO] starting...")
pba = Application()
