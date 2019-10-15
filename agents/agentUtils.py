import numpy as np


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
