import PySimpleGUI as sg
import cv2
import numpy as np


class Application:
    def __init__(self):

        sg.theme('BluePurple')

        self.graph_size = (600, 400)
        self.ORIGINAL_KEY = '-ORIGINAL-'
        self.PROCESSED_KEY = '-PROCESSED-'
        self.PARTICLE_SIZE_KEY = '-PARTICLE-SIZE-'
        self.FILENAME_KEY = '-FILENAME-'
        self.SCALE_SLIDER_KEY = '-SCALE-SLIDER-KEY-'
        self.IMAGE_SLIDER_KEY = '-IMAGE-SLIDER-KEY-'
        self.SCALE_SIZE_KEY = '-SCALE-SIZE-KEY-'
        self.DONE_KEY = '-DONE-'

        layout = self.get_layout()

        window = sg.Window('Particle Sizing', layout)
        orig_graph = window[self.ORIGINAL_KEY]
        processed_graph = window[self.PROCESSED_KEY]
        particle_size = window[self.PARTICLE_SIZE_KEY]
        scale_slider = window[self.SCALE_SLIDER_KEY]
        image_slider = window[self.IMAGE_SLIDER_KEY]
        scale_size_input = window[self.SCALE_SIZE_KEY]

        orig_image_id = None
        orig_image = None
        dragging = False
        scale_size_pixels = None
        start_point = end_point = prior_rect_id = None

        window.finalize()

        scale_slider.bind('Left', '+DEC+')
        scale_slider.bind('Right', '+INC+')

        image_slider.bind('Left', '+DEC+')
        image_slider.bind('Right', '+INC+')

        while True:  # Event Loop
            event, values = window.read()
            print("event: ", event, '.', 'values:', values, ',')
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == self.FILENAME_KEY:
                if values[self.FILENAME_KEY] is None or len(values[self.FILENAME_KEY]) < 1:
                    continue
                orig_image = cv2.imread(
                    values[self.FILENAME_KEY], cv2.IMREAD_COLOR)
                img_bytes, location = self.scale_img_to_graph(
                    orig_image, orig_graph)
                if orig_image_id:
                    orig_graph.delete_figure(orig_image_id)
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)
            elif event.endswith('+DEC+'):
                if event.startswith(self.SCALE_SLIDER_KEY):
                    scale_slider.update(
                        value=max(0, values[self.SCALE_SLIDER_KEY]-1))
                elif event.startswith(self.IMAGE_SLIDER_KEY):
                    image_slider.update(
                        value=max(0, values[self.IMAGE_SLIDER_KEY]-1))
                current_focus_slider = scale_slider
            elif event.endswith('+INC+'):
                if event.startswith(self.SCALE_SLIDER_KEY):
                    scale_slider.update(
                        value=min(255, values[self.SCALE_SLIDER_KEY]+1))
                elif event.startswith(self.IMAGE_SLIDER_KEY):
                    image_slider.update(
                        value=min(255, values[self.IMAGE_SLIDER_KEY]+1))
                current_focus_slider = scale_slider
            elif event == self.DONE_KEY:
                # get region of interest
                if prior_rect_id is None or orig_image is None:
                    continue
                roi, ys, xs = self.get_region_of_interest(
                    orig_graph, prior_rect_id, orig_image)

                # Blur image
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                _, thr = cv2.threshold(
                    blur, values[self.SCALE_SLIDER_KEY], 255, cv2.THRESH_BINARY_INV)

                # get connected components
                num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    thr, connectivity=4)

                # get the lengths and width of the connected components (0 is background so exclude it)
                lengths_and_widths = np.stack(
                    (stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]))

                # find the longest/widest one (assume that one is the image scale)
                # add 1 because we ignored background
                largest_cc = (np.argmax(lengths_and_widths) %
                              lengths_and_widths.shape[1]) + 1

                # highlight it bright red on the left graph
                # In HSV 179 is red
                label_hue = np.uint8(np.where(labels == largest_cc, 179, 0))
                blank_ch = 255 * np.ones_like(label_hue)
                labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

                # Converting cvt to BGR
                labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
                labeled_img[label_hue == 0] = 0

                # Now create a mask of scale and create its inverse mask also
                img2gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # Now black-out the area of scale in ROI
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # Take only region of scale from scale image.
                img2_fg = cv2.bitwise_and(labeled_img, labeled_img, mask=mask)

                # Put outlined scale in ROI and modify the main image
                dst = cv2.add(img1_bg, img2_fg)
                new_img = orig_image.copy()
                new_img[ys[0]:ys[1], xs[0]:xs[1]] = dst

                img_bytes, location = self.scale_img_to_graph(
                    new_img, orig_graph)
                orig_graph.delete_figure(orig_image_id)
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)

                # calculate real-life size of pixel
                scale_size_pixels = np.max(lengths_and_widths)
                pixel_size_in_microns = (self.get_scale_mm(
                    values[self.SCALE_SIZE_KEY]) * 1000) / scale_size_pixels

                particle = orig_image.copy()
                # (self, particle, thresh, ys, xs, pixel_size_in_microns, particle_size, processed_graph):
                self.draw_processed_particle(
                    particle, values[self.IMAGE_SLIDER_KEY], ys, xs, pixel_size_in_microns, particle_size, processed_graph)
            elif event == self.ORIGINAL_KEY:
                x, y = values[self.ORIGINAL_KEY]
                if not dragging:
                    start_point = (x, y)
                    dragging = True
                else:
                    end_point = (x, y)
                if prior_rect_id:
                    orig_graph.delete_figure(prior_rect_id)
                if None not in (start_point, end_point):
                    prior_rect_id = orig_graph.draw_rectangle(
                        start_point, end_point, line_color='red')
            elif event == self.SCALE_SLIDER_KEY:
                scale_slider.set_focus(True)
                if prior_rect_id is None or orig_image is None:
                    continue
                img_bytes, location = self.draw_thresholded_scale(
                    values[self.SCALE_SLIDER_KEY], orig_graph, orig_image, orig_image_id, prior_rect_id)
                orig_graph.delete_figure(orig_image_id)
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)
            elif event == self.IMAGE_SLIDER_KEY:
                image_slider.set_focus(True)
                current_focus_slider = image_slider
                if prior_rect_id is None or orig_image is None:
                    continue
                roi, ys, xs = self.get_region_of_interest(
                    orig_graph, prior_rect_id, orig_image)

                pixel_size_in_microns = (self.get_scale_mm(
                    values[self.SCALE_SIZE_KEY]) * 1000) / scale_size_pixels

                particle = orig_image.copy()
                self.draw_processed_particle(
                    particle, values[self.IMAGE_SLIDER_KEY], ys, xs, pixel_size_in_microns, particle_size, processed_graph)
            elif event == self.SCALE_SIZE_KEY:
                inputVal = self.get_scale_mm(values[self.SCALE_SIZE_KEY])
                if inputVal is None:
                    scale_size_input.update(background_color='red')
                else:
                    scale_size_input.update(background_color='#E0F5FF')

                if prior_rect_id is None or orig_image is None:
                    continue
                roi, ys, xs = self.get_region_of_interest(
                    orig_graph, prior_rect_id, orig_image)

                pixel_size_in_microns = (self.get_scale_mm(
                    values[self.SCALE_SIZE_KEY]) * 1000) / scale_size_pixels

                particle = orig_image.copy()
                self.draw_processed_particle(
                    particle, values[self.IMAGE_SLIDER_KEY], ys, xs, pixel_size_in_microns, particle_size, processed_graph)
            elif event == "Left:37" or event == "Right:39":
                if event == "Left:37":
                    val -= 1
                else:
                    val += 1

                if current_focus_slider is not None:
                    cur_val = values[current_focus_slider.key]
                    current_focus_slider.update(value=cur_val + val)

            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False

                img_bytes, location = self.draw_thresholded_scale(
                    values[self.SCALE_SLIDER_KEY], orig_graph, orig_image, orig_image_id, prior_rect_id)
                orig_graph.delete_figure(orig_image_id)
                orig_image_id = orig_graph.draw_image(
                    data=img_bytes, location=location)

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

    def get_scale_mm(self, inputVal):
        try:
            return float(inputVal)
        except:
            return None

    def draw_thresholded_scale(self, thresh, graph, image, image_id, box_id):
        if box_id is None or image is None:
            return

        roi, ys, xs = self.get_region_of_interest(
            graph, box_id, image)

        # Blur image
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        _, thr = cv2.threshold(
            roi_gray, thresh, 255, cv2.THRESH_BINARY)

        thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)

        new_img = image.copy()
        new_img[ys[0]:ys[1], xs[0]:xs[1]] = thr

        return self.scale_img_to_graph(
            new_img, graph)

    def draw_processed_particle(self, particle, thresh, ys, xs, pixel_size_in_microns, particle_size, processed_graph):
        if pixel_size_in_microns is None:
            return
        # remove highlighted area
        particle[ys[0]:ys[1], xs[0]:xs[1]] = 255

        # apply threshold
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        particle = cv2.filter2D(particle, -1, kernel)
        particle_blur = cv2.cvtColor(particle, cv2.COLOR_BGR2GRAY)
        particle_blur = cv2.GaussianBlur(particle_blur, (5, 5), 0)
        thresh = int(thresh)
        particle_thresh = cv2.threshold(
            particle_blur, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        # particle_thresh = cv2.adaptiveThreshold(
        #     particle_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, thresh)
        num_particle_pixels = cv2.countNonZero(particle_thresh)
        particle_thresh[ys[0] - 2:ys[1] + 2, xs[0] - 2:xs[1] + 2] = 0

        # calculate area
        particle_area = (pixel_size_in_microns * pixel_size_in_microns *
                         num_particle_pixels) / (1000000)
        particle_area = round(particle_area, 4)

        # reflect changes in gui
        particle_size.update(
            value=f' , then particle size is: {particle_area} mm2')
        particle_bytes, location = self.scale_img_to_graph(
            particle_thresh, processed_graph)
        processed_graph.erase()
        processed_graph.draw_image(
            data=particle_bytes, location=location)

    def get_layout(self):
        return [[sg.Text('1. Please select a photo:'),
                 sg.InputText(size=(50, 1), key=self.FILENAME_KEY,
                              enable_events=True, readonly=True),
                 sg.FileBrowse()],
                [sg.Text('2. Please highlight the text and scale with your mouse.\n   Adjust the threshold, then press done.'),
                 sg.Button('Done', key=self.DONE_KEY)],
                [sg.Text('Scale Threshold'), sg.Slider(range=(0, 255), default_value=127,
                                                       orientation='horizontal', key=self.SCALE_SLIDER_KEY, enable_events=True)],
                [sg.Text('Image Threshold'), sg.Slider(range=(0, 255), default_value=127,
                                                       orientation='horizontal', key=self.IMAGE_SLIDER_KEY, enable_events=True)],
                [self.create_original_image(), sg.VSep(), self.create_cleaned_image()],
                [sg.Text('3. If the scale is '), sg.InputText(default_text='1.00', size=(5, 1), enable_events=True, key=self.SCALE_SIZE_KEY), sg.Text('mm'), sg.InputText(default_text='mm , then particle size is: ',
                                                                                                                                                                          key=self.PARTICLE_SIZE_KEY, readonly=True)],
                [sg.Text('4. Choose another image.')]
                ]

    def create_original_image(self):
        return sg.Graph(key=self.ORIGINAL_KEY, canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size,
                        border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)

    def create_cleaned_image(self):
        return sg.Graph(key=self.PROCESSED_KEY, canvas_size=self.graph_size, graph_bottom_left=(0, 0),
                        graph_top_right=self.graph_size, background_color='lightblue',
                        border_width=1)


# start the app
print("[INFO] starting...")
pba = Application()
