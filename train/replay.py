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

for step in range(length):
    image.set_data(visuals[step, 0, :, :, :])
    fig.suptitle('Step = ' + str(step))
    #TODO add an arrow to indicate the action
    fig.canvas.draw()
    fig.canvas.flush_events()
    #TODO add a keyboard listner to control, like using left and right to go forward and backward


# last frame
image.set_data(visuals[length, 0, :, :, :])
fig.suptitle('Step = ' + length)
fig.canvas.draw()
fig.canvas.flush_events()
