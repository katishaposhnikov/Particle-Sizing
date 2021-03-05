from typing import Tuple

import PySimpleGUI as sg
import cv2


class Graph:
    def __init__(self, key: str, graph_size: Tuple[int, int]):
        self.graph = sg.Graph(key=key, canvas_size=graph_size, graph_bottom_left=(0, graph_size[1]),
                              graph_top_right=(graph_size[0], 0),
                              border_width=1, change_submits=True, background_color='lightblue', drag_submits=True)
        self.graph_scale = None
        self.selection_id = None
        self.region_of_interest_id = None

        self.start_point = None
        self.end_point = None
        self.dragging = False

    def layout(self):
        return [self.graph]

    def scale_to_image(self, img):
        self.graph.change_coordinates((0, img.shape[1]), (img.shape[0], 0))
        self.graph_scale = min(self.graph.get_size()[1] / img.shape[0],
                               self.graph.get_size()[0] / img.shape[1])

    def draw_particle(self, particle):
        self.scale_to_image(particle.img)
        width = int(particle.img.shape[1] * self.graph_scale)
        height = int(particle.img.shape[0] * self.graph_scale)
        self.graph.set_size(size=(width, height))
        resized = cv2.resize(particle.img, (width, height))
        img_bytes = cv2.imencode('.png', resized)[1].tobytes()
        self.graph.draw_image(data=img_bytes, location=(0, 0))

    def draw_roi(self, roi):
        pass

    def handle_mouse_event(self, x, y):
        if not self.dragging:
            self.start_point = (x, y)
            self.dragging = True
        else:
            self.end_point = (x, y)

    def handle_mouse_up_event(self):
        self.start_point, self.end_point = None, None  # enable grabbing a new rect
        self.dragging = False

    def erase(self):
        self.graph.erase()
