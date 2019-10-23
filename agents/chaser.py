from skimage import measure
import numpy as np
import AgentConstants
from bresenham import bresenham


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_is_color = None

    def chase(self):
        self.newest_is_color = self.agent.is_color
        self.chase_internal(self.newest_is_color)

    def chase_in_dark(self):
        imaginary_is_color = self.newest_is_color # TODO enhance
        self.chase_internal(imaginary_is_color)

    def chase_internal(self, is_color):

        labels, label_num = measure.label(input=is_color, background=False, return_num=True, connectivity=1)
        sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
        target_label = np.argmax(sizes) + 1
        target_center = np.array(np.where(labels == target_label)).mean(axis=1).astype(np.int)
        target_size = sizes[target_label - 1]

        line_idx = tuple(np.array(list(bresenham(AgentConstants.standpoint[0], AgentConstants.standpoint[1],
                         target_center[0], target_center[1]))))

        while not self.agent.is_red[line_idx].any():
            pass

        self.generate_action(line_idx)