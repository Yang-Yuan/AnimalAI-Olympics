import agentUtils
import numpy as np

# Task-related constants
predefined_colors = {"green": [0.506, 0.749, 0.255],
                     "brown": [0.471, 0.337, 0.0471],
                     "red": [0.722, 0.196, 0.196],
                     "orange": [1, 0.675, 0.282],
                     "box_dark": [0.196, 0.165, 0.133],
                     "box_light": [0.318, 0.267, 0.22],
                     "UL": [0.435, 0.367, 0.2],
                     "yellow": [0.733, 0.651, 0.506]}
predefined_colors_h = {k: agentUtils.toHue(np.array(v, ndmin=3))[0, 0] \
                       for (k, v) in predefined_colors.items()}

# Environmental constants
resolution = 84
center_of_view = [resolution / 2, resolution / 2]
default_test_length = 1000

# Perception limits of colors
color_diff_limit = 0.075  # TODO different limits for different object
road_mask = np.full(shape = (resolution, resolution), fill_value = False, dtype = bool)
for delta, ii in zip(np.arange(resolution / 2, dtype = int),
                     np.arange(start = resolution - 1, stop = (resolution - 1) / 2, step = -1, dtype = int)):
    for jj in np.arange(start = 0 + delta, stop = resolution - delta, dtype = int):
        road_mask[ii, jj] = True


# Control constants
aim_error_limit = 5
size_limit = 5
hl = 2
pirouette_step_limit = 60
roam_step_limit = 10
red_pixel_on_road_limit = 882

# actions
right = [0, 1]
left = [0, 2]
forward = [1, 0]
backward = [2, 0]

