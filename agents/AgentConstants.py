import agentUtils
import numpy as np

# Task-related constants
predefined_colors = {"green": [0.506, 0.749, 0.255],
                     "brown": [0.471, 0.337, 0.0471],
                     "red": [0.722, 0.196, 0.196],
                     "orange": [1, 0.675, 0.282],
                     "box_dark": [0.196, 0.165, 0.133],
                     "box_light": [0.318, 0.267, 0.22],
                     "UL": [0.435, 0.367, 0.2]}
predefined_colors_h = {k: agentUtils.toHue(np.array(v, ndmin=3))[0, 0] \
                       for (k, v) in predefined_colors.items()}

# Environmental constants
resolution = 84
center_of_view = [resolution / 2, resolution / 2]
default_test_length = 1000

# Perception limits of colors
color_diff_limit = 0.075  # TODO different limits for different object

# Control constants
aim_error_limit = 5
size_limit = 5
hl = 2
pirouette_step_limit = 60