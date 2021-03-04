from typing import Tuple

import PySimpleGUI as sg


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
