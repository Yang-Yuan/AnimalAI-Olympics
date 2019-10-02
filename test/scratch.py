from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np

n = 12
l = 256
np.random.seed(1)
im = np.zeros((l, l))
points = l * np.random.random((2, n ** 2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = filters.gaussian(im, sigma= l / (4. * n))
blobs = im > 0.7 * im.mean()

all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0)

plt.figure(figsize=(9, 3.5))
plt.subplot(131)
plt.imshow(blobs, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels, cmap='nipy_spectral')
plt.axis('off')
plt.subplot(133)
plt.imshow(blobs_labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# img = mpimg.imread('Figure_1.png')
# imgplot = plt.imshow(img)
# plt.show()
#
# tmp = np.zeros((84, 84))
# for ii in range(84):
#     for jj in range(84):
#         tmp[ii, jj] = cosine(obs[ii, jj], Agent.green)
#
# tmp.min()
#
#
# def toHue(rgb):
#     ind_min = rgb.argmin()
#     ind_max = rgb.argmax()
#     diff = rgb[ind_max] - rgb[ind_min]
#
#     if 0 == ind_max:
#         h = (rgb[1] - rgb[2]) / diff
#     elif 1 == ind_max:
#         h = 2 + (rgb[2] - rgb[0]) / diff
#     else:
#         h = 4 + (rgb[0] - rgb[1]) / diff
#
#     if h < 0:
#         h += 6
#
#     return h
#
# green = np.array([0.506, 0.749, 0.255])