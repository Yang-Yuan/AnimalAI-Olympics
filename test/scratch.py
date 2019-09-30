import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('Figure_1.png')
imgplot = plt.imshow(img)
plt.show()

tmp = np.zeros((84, 84))
for ii in range(84):
    for jj in range(84):
        tmp[ii, jj] = cosine(obs[ii, jj], Agent.green)

tmp.min()


def toHue(rgb):
    ind_min = rgb.argmin()
    ind_max = rgb.argmax()
    diff = rgb[ind_max] - rgb[ind_min]

    if 0 == ind_max:
        h = (rgb[1] - rgb[2]) / diff
    elif 1 == ind_max:
        h = 2 + (rgb[2] - rgb[0]) / diff
    else:
        h = 4 + (rgb[0] - rgb[1]) / diff

    if h < 0:
        h += 6

    return h

green = np.array([0.506, 0.749, 0.255])