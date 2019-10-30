import numpy as np
from bresenham import bresenham
from skimage import measure


def toHue(rgb):
    out_h = np.zeros(rgb.shape[:-1])

    out_v = rgb.max(-1)  # max
    delta = rgb.ptp(-1)  # max - min

    old_settings = np.seterr(invalid='ignore')

    # red is max
    idx = (rgb[:, :, 0] == out_v)
    out_h[idx] = (rgb[idx, 1] - rgb[idx, 2]) / delta[idx]

    # green is max
    idx = (rgb[:, :, 1] == out_v)
    out_h[idx] = 2. + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]

    # blue is max
    idx = (rgb[:, :, 2] == out_v)
    out_h[idx] = 4. + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]

    # normalization
    out_h = (out_h / 6.) % 1.
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)

    # remove NaN
    out_h[np.isnan(out_h)] = 0

    return out_h


def render_line_segments(critical_points):
    line_idx = list()
    for kk in np.arange(len(critical_points) - 1):
        new_list = list(bresenham(critical_points[kk][0], critical_points[kk][1],
                                  critical_points[kk + 1][0], critical_points[kk + 1][1]))
        if kk == len(critical_points) - 2:
            line_idx = line_idx + new_list
        else:
            line_idx = line_idx + new_list[: -1]

    return line_idx


def is_color_significant(is_color, significant_size):
    if is_color.any():
        labels, label_num = measure.label(input=is_color,
                                          background=False,
                                          return_num=True, connectivity=1)
        sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
        return max(sizes) > significant_size
    else:
        return False
