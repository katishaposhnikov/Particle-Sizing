import PySimpleGUI as sg
import cv2
import numpy as np

################ CONSTANTS ################

ORIGINAL_KEY = '-ORIGINAL-'
PROCESSED_KEY = '-PROCESSED-'
PARTICLE_SIZE_KEY = '-PARTICLE-SIZE-'
FILENAME_KEY = '-FILENAME-'
SCALE_SLIDER_KEY = '-SCALE-SLIDER-KEY-'
IMAGE_SLIDER_KEY = '-IMAGE-SLIDER-KEY-'
SCALE_SIZE_KEY = '-SCALE-SIZE-KEY-'
MASK_RADIO_KEY = '-MASK-RADIO-KEY-'
PARTICLE_RADIO_KEY = '-PARTICLE-RADIO-KEY-'
BACKGROUND_RADIO_KEY = 'BACKGROUND-RADIO-KEY-'
AUTOMATIC_KEY = '-AUTOMATIC-KEY-'
GRAPH_SIZE = (400, 300)


################ CONSTANTS ################


class RegionOfInterest:
    def __init__(self, imgbytes=None, y_bounds=None, x_bounds=None):
        # the raw image bytes returned by cv2.imencode('.png', resize)[1].tobytes()
        # ready to be accepted by graph.draw_image
        self.imgbytes = imgbytes

        # the 'row' or 'y' bounds from the numpy array that is the original image
        self.y_bounds = y_bounds

        # the 'column' or 'x' bounds from the numpy array that is the original image
        self.x_bounds = x_bounds

        # together cv2.imread('image.png')[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]
        # gives the region of interest

    def top_left(self):
        return self.x_bounds[0], self.y_bounds[0]


def create_original_image():
    return sg.Graph(key=ORIGINAL_KEY, canvas_size=GRAPH_SIZE, graph_bottom_left=(0, GRAPH_SIZE[1]),
                    graph_top_right=(GRAPH_SIZE[0], 0),
                    border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)


def create_processed_image():
    return sg.Graph(key=PROCESSED_KEY, canvas_size=GRAPH_SIZE, graph_bottom_left=(0, GRAPH_SIZE[1]),
                    graph_top_right=(GRAPH_SIZE[0], 0), background_color='lightblue',
                    border_width=1)


def get_layout():
    return [
        [sg.Column([[sg.Text('1. Please select a photo:'),
                     sg.InputText(size=(50, 1), key=FILENAME_KEY,
                                  enable_events=True, readonly=True),
                     sg.FileBrowse()],
                    [sg.Text(
                        '2. Please highlight the text and scale with your mouse.')],
                    [sg.Column([[sg.Column([[sg.Text('Scale Threshold'), sg.Slider(range=(0, 255), default_value=127,
                                                                                   orientation='horizontal',
                                                                                   key=SCALE_SLIDER_KEY,
                                                                                   enable_events=True)],
                                            [sg.Text('Image Threshold'), sg.Slider(range=(0, 255), default_value=127,
                                                                                   orientation='horizontal',
                                                                                   key=IMAGE_SLIDER_KEY,
                                                                                   enable_events=True)]]),
                                 sg.Button('Automatic', key=AUTOMATIC_KEY, font=('Helvetica', 30)), ],

                                [create_original_image()],
                                ], element_justification='center',
                               vertical_alignment='bottom'),
                     sg.Column(
                         [[sg.Radio(text='Mask', group_id='overlay', enable_events=True, default=True,
                                    key=MASK_RADIO_KEY)],
                          [sg.Radio(text='Particle', group_id='overlay', enable_events=True,
                                    key=PARTICLE_RADIO_KEY)],
                          [sg.Radio(text='Background', group_id='overlay', enable_events=True,
                                    key=BACKGROUND_RADIO_KEY)], [create_processed_image()]],
                         element_justification='left', vertical_alignment='bottom'
                     )],
                    [sg.Text('3. If the scale is '),
                     sg.InputText(default_text='1.00', size=(5, 1), enable_events=True, key=SCALE_SIZE_KEY),
                     sg.Text('mm, then the particle is: '), sg.InputText(default_text='',
                                                                         key=PARTICLE_SIZE_KEY, readonly=True)],
                    [sg.Text('4. Choose another image.')]])],
    ]


class Application:
    def __init__(self):
        sg.theme('BluePurple')

        window = sg.Window('Particle Sizing', get_layout(), resizable=True)
        self.orig_graph = window[ORIGINAL_KEY]
        self.processed_graph = window[PROCESSED_KEY]
        self.particle_size = window[PARTICLE_SIZE_KEY]
        self.scale_slider = window[SCALE_SLIDER_KEY]
        self.image_slider = window[IMAGE_SLIDER_KEY]
        self.scale_size_input = window[SCALE_SIZE_KEY]
        self.current_focus_slider = None

        self.orig_image = None  # np.zeros((1, 1, 4), np.uint8)
        self.processed_image = None  # np.zeros((1, 1, 4), np.uint8)
        self.region_of_interest = None
        self.graph_scale_factor = None
        self.region_of_interest_id = None
        self.prior_rect_id = None
        self.scale_size_pixels = None
        self.scale_size_mm = None
        self.radio_option = MASK_RADIO_KEY

        self.start_point = None
        self.end_point = None
        self.dragging = False

        self.pixel_size_in_microns = None
        self.scale_size_mm = None
        self.scale_size_pixels = None
        self.num_particle_pixels = None

        self.should_redraw_original = False
        self.should_redraw_roi = False
        self.should_redraw_processed = False

        window.finalize()

        self.scale_slider.bind('Left', '+DEC+')
        self.scale_slider.bind('Right', '+INC+')

        self.image_slider.bind('Left', '+DEC+')
        self.image_slider.bind('Right', '+INC+')

        self.scale_size_input_original_bg_color = self.scale_size_input.BackgroundColor

        while True:  # Event Loop
            self.should_redraw_original = self.should_redraw_roi = self.should_redraw_processed = False
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == FILENAME_KEY:
                if values[FILENAME_KEY] is None or len(values[FILENAME_KEY]) < 1:
                    continue
                self.orig_image = cv2.imread(values[FILENAME_KEY], cv2.IMREAD_COLOR)
                self.orig_graph.change_coordinates((0, self.orig_image.shape[0]), (self.orig_image.shape[1], 0))
                self.processed_graph.change_coordinates((0, self.orig_image.shape[1]), (self.orig_image.shape[0], 0))
                self.graph_scale_factor = min(GRAPH_SIZE[1] / self.orig_image.shape[0],
                                              GRAPH_SIZE[0] / self.orig_image.shape[1])
                self.region_of_interest = None
                self.should_redraw_original = True
            elif event == ORIGINAL_KEY:
                x, y = values[ORIGINAL_KEY]
                if not self.dragging:
                    self.start_point = (x, y)
                    self.dragging = True
                else:
                    self.end_point = (x, y)
                self.should_redraw_roi = True
            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                if values[FILENAME_KEY] is not None and len(
                        values[FILENAME_KEY]) > 0 and self.start_point and self.end_point:
                    self.create_region_of_interest()
                    self.create_processed_particle()
                    self.should_redraw_roi = True
                    self.should_redraw_processed = True
                self.start_point, self.end_point = None, None  # enable grabbing a new rect
                self.dragging = False
            elif event.endswith('+DEC+') or event.endswith('+INC+'):
                self.update_sliders(event, values)
            elif event == IMAGE_SLIDER_KEY:
                self.image_slider.set_focus(True)
                self.current_focus_slider = self.image_slider
                self.create_processed_particle(thresh=values[IMAGE_SLIDER_KEY])
                self.should_redraw_processed = True
            elif event == SCALE_SLIDER_KEY:
                self.scale_slider.set_focus(True)
                self.current_focus_slider = self.scale_slider
                self.create_region_of_interest(thresh=values[SCALE_SLIDER_KEY])
                self.create_processed_particle(thresh=values[IMAGE_SLIDER_KEY])
                self.should_redraw_roi = True
                self.should_redraw_processed = True
            elif event == MASK_RADIO_KEY or event == PARTICLE_RADIO_KEY or event == BACKGROUND_RADIO_KEY:
                self.radio_option = event
                self.should_redraw_processed = True
            elif event == AUTOMATIC_KEY:
                self.create_region_of_interest()
                self.create_processed_particle()
                self.should_redraw_roi = True
                self.should_redraw_processed = True
            elif event == SCALE_SIZE_KEY:
                self.update_particle_size()

            self.redraw()

    def redraw(self):
        if self.should_redraw_original:
            self.redraw_original()
        if self.should_redraw_roi:
            self.redraw_roi()
        if self.should_redraw_processed:
            self.redraw_processed()

    def redraw_original(self):
        self.orig_graph.erase()

        if self.orig_image is not None:
            img_bytes = cv2.imencode('.png', cv2.resize(self.orig_image, (
                int(self.orig_image.shape[1] * self.graph_scale_factor),
                int(self.orig_image.shape[0] * self.graph_scale_factor))))[1].tobytes()
            self.orig_graph.draw_image(data=img_bytes, location=(0, 0))

    def redraw_roi(self):
        if self.prior_rect_id:
            self.orig_graph.delete_figure(self.prior_rect_id)
            self.prior_rect_id = None
        if self.region_of_interest_id:
            self.orig_graph.delete_figure(self.region_of_interest_id)

        if None not in (self.start_point, self.end_point):
            self.prior_rect_id = self.orig_graph.draw_rectangle(
                self.start_point, self.end_point, line_color='red')
        elif self.region_of_interest is not None:
            self.region_of_interest_id = self.orig_graph.draw_image(
                data=self.region_of_interest.imgbytes,
                location=self.region_of_interest.top_left())

    def redraw_processed(self):
        self.processed_graph.erase()

        if self.processed_image is not None and self.orig_image is not None:
            if self.radio_option == BACKGROUND_RADIO_KEY:
                img_to_draw = cv2.add(self.orig_image, cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
            elif self.radio_option == PARTICLE_RADIO_KEY:
                img_to_draw = cv2.bitwise_and(self.orig_image, self.orig_image, mask=self.processed_image)
            else:
                img_to_draw = self.processed_image

            bytes_to_draw = cv2.imencode('.png', cv2.resize(img_to_draw, (
                int(self.processed_image.shape[1] * self.graph_scale_factor),
                int(self.processed_image.shape[0] * self.graph_scale_factor)),
                                                            interpolation=cv2.INTER_AREA))[1].tobytes()
            self.processed_graph.draw_image(data=bytes_to_draw, location=(0, 0))

    def create_region_of_interest(self, thresh=None):
        if self.prior_rect_id is not None:
            upper_left = min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1])
            lower_right = max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1])
            ys = (upper_left[1], lower_right[1])
            xs = (upper_left[0], lower_right[0])
        elif self.region_of_interest is not None:
            ys = self.region_of_interest.y_bounds
            xs = self.region_of_interest.x_bounds
        else:
            return

        roi = self.orig_image[ys[0]:ys[1], xs[0]:xs[1]]
        if 0 not in roi.shape:
            annotated_roi = self.extract_scale(roi, thresh)
            self.region_of_interest = RegionOfInterest(
                cv2.imencode('.png',
                             cv2.resize(annotated_roi,
                                        (int(annotated_roi.shape[1] * self.graph_scale_factor),
                                         int(annotated_roi.shape[0] * self.graph_scale_factor),
                                         ),
                                        ),
                             )[1].tobytes()
                , ys, xs)
            return True

    def extract_scale(self, roi, thresh):
        # Blur image
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        if thresh:
            _, thr_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
        else:
            thr_val, thr_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.scale_slider.update(value=thr_val)

        # get connected components
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thr_img, connectivity=4)

        if num_components <= 1:
            # we couldn't find anything to extract (there is just one giant blob) so return and wait for
            # the user to set better image parameters
            return roi

        # get the lengths and width of the connected components (0 is background so exclude it)
        lengths_and_widths = np.stack(
            (stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]))

        # find the longest/widest one (assume that one is the image scale)
        # add 1 because we ignored background
        largest_cc = (np.argmax(lengths_and_widths) %
                      lengths_and_widths.shape[1]) + 1

        self.scale_size_pixels = np.max(lengths_and_widths)

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

        # Return the outlined scale in roi
        dst = cv2.add(img1_bg, img2_fg)
        return dst

    def create_processed_particle(self, thresh=None):
        if self.region_of_interest is None:
            return
        # remove the scale area
        particle = self.orig_image.copy()
        ys = self.region_of_interest.y_bounds
        xs = self.region_of_interest.x_bounds
        particle[ys[0]:ys[1], xs[0]:xs[1]] = 255
        # apply thresholding
        particle_gray = cv2.cvtColor(particle, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(particle_gray, (5, 5), 0)
        if thresh:
            _, particle_thresh = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
        else:
            particle_thr_val, particle_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # update the ui to reflect the automatically chosen threshold
            self.image_slider.update(value=particle_thr_val)

        # calculate number of pixels
        self.num_particle_pixels = cv2.countNonZero(particle_thresh)

        self.update_particle_size()

        # create the particle mask
        self.processed_image = particle_thresh

    def get_scale_mm(self):
        input_val = self.scale_size_input.get()
        if input_val is None or len(input_val) == 0:
            return None
        try:
            return float(input_val)
        except ValueError:
            return None

    def update_sliders(self, event, values):
        if event.endswith('+DEC+'):
            if event.startswith(SCALE_SLIDER_KEY):
                self.scale_slider.update(
                    value=max(0, values[SCALE_SLIDER_KEY] - 1))
                self.current_focus_slider = self.scale_slider
                self.create_region_of_interest(max(0, values[SCALE_SLIDER_KEY] - 1))
                self.should_redraw_roi = True
            elif event.startswith(IMAGE_SLIDER_KEY):
                self.image_slider.update(
                    value=max(0, values[IMAGE_SLIDER_KEY] - 1))
                self.current_focus_slider = self.image_slider
                self.should_redraw_processed = True
                self.create_processed_particle(max(0, values[IMAGE_SLIDER_KEY] - 1))

        elif event.endswith('+INC+'):
            if event.startswith(SCALE_SLIDER_KEY):
                self.scale_slider.update(
                    value=min(255, values[SCALE_SLIDER_KEY] + 1))
                self.current_focus_slider = self.scale_slider
                self.should_redraw_roi = True
                self.create_region_of_interest(min(255, values[SCALE_SLIDER_KEY] + 1))
            elif event.startswith(IMAGE_SLIDER_KEY):
                self.image_slider.update(
                    value=min(255, values[IMAGE_SLIDER_KEY] + 1))
                self.current_focus_slider = self.image_slider
                self.should_redraw_processed = True
                self.create_processed_particle(min(255, values[IMAGE_SLIDER_KEY] + 1))

    def update_particle_size(self):
        # calculate real-life size of pixel
        scale_mm = self.get_scale_mm()
        if scale_mm is None:
            self.scale_size_input.update(background_color='#FF5252')
            return
        else:
            self.scale_size_input.update(background_color=self.scale_size_input_original_bg_color)
        if self.scale_size_pixels is None or self.num_particle_pixels is None:
            # we haven't uploaded a picture yet so don't do anything
            return
        pixel_size_in_microns = (scale_mm * 1000) / self.scale_size_pixels

        # calculate area of particle
        particle_area = (pixel_size_in_microns * pixel_size_in_microns *
                         self.num_particle_pixels) / 1000000
        particle_area = round(particle_area, 4)

        # update size ui
        self.particle_size.update(value=f'{particle_area} mm2')


# start the app
print("[INFO] starting...")
app = Application()
