import agentUtils
import numpy as np

# Task-related constants
predefined_colors = {"green": [0.506, 0.749, 0.255],
                     "brown": [0.267, 0.188, 0.0118],
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
memory_size = 60
green_tolerance = 0.06510537801131308
brown_tolerance = 0.001
red_tolerance = 0.001
# red_tolerance = 0.03996598639455783
# orange_tolerance = 0.003284771042431224
# box_dark_tolerance = 0.002362055933484493
# box_light_tolerance = 0.002362055933484493
# UL_tolerance = 0.002229215708910319
yellow_tolerance = 0.0037600991284689814

# road_mask = np.full(shape = (resolution, resolution), fill_value = False, dtype = bool)
# for delta, ii in zip(np.repeat(np.arange(resolution / 4), 2),
#                      np.arange(start = resolution - 1, stop = (resolution - 1) / 2, step = -1, dtype = int)):
#     for jj in np.arange(start = 0 + delta, stop = resolution - delta, dtype = int):
#         road_mask[ii, jj] = True

road_mask = np.full(shape = (resolution, resolution), fill_value = False, dtype = bool)
for delta, ii in zip(np.repeat(np.arange(resolution / 2, dtype = int)[::-1], 2),
                     np.arange(start = 0, stop = resolution, step = 1, dtype = int)):
    for jj in np.arange(start = 0 + delta, stop = resolution - delta, dtype = int):
        road_mask[ii, jj] = True

frame_mask = np.full(shape = (resolution, resolution), fill_value = False, dtype = bool)
frame_mask[0, :] = True
frame_mask[resolution - 1, :] = True
frame_mask[:, 0] = True
frame_mask[:, resolution - 1] = True
frame_idx = np.argwhere(frame_mask)
all_false = np.full((resolution, resolution), False)

# Control constants
aim_error_limit = 5
size_limit = 5
hl = 2
pirouette_step_limit = 60
not_seeing_target_step_limit = 60
roam_step_limit = 10
minimal_dist_to_in_accessible = resolution / 2
standpoint = [83, 41]
path_consistent_ratio = 0.5

# actions
taxi = [0, 0]
right = [0, 1]
left = [0, 2]
forward = [1, 0]
backward = [2, 0]
forward_left = [1, 2]
forward_right = [1, 1]


