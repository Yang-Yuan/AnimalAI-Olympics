from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import constants
import time

Tk().withdraw()
fileName = askopenfilename()

npzFile = np.load(fileName)

visuals = npzFile['visuals']
actions = npzFile['actions']

length = actions.shape[0]

plt.ion()
fig, ax = plt.subplots()
image = ax.imshow(np.zeros((constants.resolution, constants.resolution, constants.n_channels)))


def getXY(action):
    x = constants.resolution / 2
    y = constants.resolution / 2

    if 0 != action[0]:
        y += (action[0] - 1.5) * constants.resolution / 2

    if 0 != action[1]:
        x += (action[1] - 1.5) * constants.resolution / 2


for step in range(length):
    image.set_data(visuals[step, 0, :, :, :])
    fig.suptitle('Step = ' + str(step))
    ax.annotate(str(actions[step, :]),
                xy = getXY(actions[step, :]),
                xytext = (constants.resolution / 2, constants.resolution / 2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    #TODO add an arrow to indicate the action
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.2)


# last frame
image.set_data(visuals[length, 0, :, :, :])
fig.suptitle('Step = ' + str(length))
fig.canvas.draw()
fig.canvas.flush_events()

