import AgentConstants
import numpy as np
import warnings
from skimage import measure

class Perception(object):

    def __init__(self, agent):
        self.agent = agent

    def perceive(self):

        if self.agent.visual_h_memory.full():
            self.agent.visual_h_memory.get()
        if self.agent.is_green_memory.full():
            self.agent.is_green_memory.get()
        if self.agent.is_brown_memory.full():
            self.agent.is_brown_memory.get()
        if self.agent.is_red_memory.full():
            self.agent.is_red_memory.get()
        if self.agent.is_orange_memory.full():
            self.agent.is_orange_memory.get()
        if self.agent.is_yellow_memory.full():
            self.agent.is_yellow_memory.get()
        if self.agent.vector_memory.full():
            self.agent.vector_memory.get()

        self.agent.is_green = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "green")) < AgentConstants.color_diff_limit
        self.agent.is_brown = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "brown")) < AgentConstants.color_diff_limit
        self.agent.is_red = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "red")) < AgentConstants.color_diff_limit
        self.agent.is_orange = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "orange")) < AgentConstants.color_diff_limit
        self.agent.is_yellow = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "yellow")) < AgentConstants.color_diff_limit
        self.agent.is_inaccessible = self.synthesize_is_inaccessible()

        self.agent.visual_h_memory.put(self.agent.obs_visual_h)
        self.agent.is_green_memory.put(self.agent.is_green)
        self.agent.is_brown_memory.put(self.agent.is_brown)
        self.agent.is_red_memory.put(self.agent.is_red)
        self.agent.is_orange_memory.put(self.agent.is_orange)
        self.agent.is_yellow_memory.put(self.agent.is_yellow)
        self.agent.vector_memory.put(self.agent.obs_vector)

    def is_front_safe(self):
        return (self.agent.is_red & AgentConstants.road_mask).sum() < AgentConstants.red_pixel_on_road_limit

    def is_static(self):
        return (self.agent.obs_vector == 0).all()

    def is_found(self):
        if self.agent.target_color == "green" and self.agent.is_green.any():
            self.agent.target_color = "green"
            self.agent.is_target_color = self.agent.is_green
            return True
        elif self.agent.target_color == "brown" and self.agent.is_brown.any():
            self.agent.target_color = "brown"
            self.agent.is_target_color = self.agent.is_brown
            return True
        else:
            return False

    def renew_target_from_panorama(self):

        if self.agent.pirouette_step_n == AgentConstants.pirouette_step_limit:

            self.agent.target_color = None
            self.agent.safest_direction = None

            if np.array(self.agent.is_brown_memory.queue)[-AgentConstants.pirouette_step_limit:].any():
                self.agent.target_color = "brown"
            elif np.array(self.agent.is_green_memory.queue)[-AgentConstants.pirouette_step_limit:].any():
                self.agent.target_color = "green"
            else:
                is_yellow = np.array(self.agent.is_yellow_memory.queue)[-AgentConstants.pirouette_step_limit:]
                if is_yellow.any():
                    self.agent.safest_direction = np.argmax(
                        [(frame & AgentConstants.road_mask).sum() for frame in self.agent.is_yellow])
                else:
                    warnings.warn("Nowhere to go, just move forward")
                    self.agent.safest_direction = 0
            print("target renewed: {} and {}".format(self.agent.target_color, self.agent.safest_direction))

    def renew_target(self):

        if self.agent.target_color == "green" and self.agent.is_brown.any():
            self.agent.target_color = "brown"
            self.agent.is_target_color = self.agent.is_brown
            return True

        if self.agent.target_color is None and self.agent.is_green.any():
            self.agent.target_color = "green"
            self.agent.is_target_color = self.agent.is_green
            return True

        if self.agent.target_color is None and self.agent.is_brown.any():
            self.agent.target_color = "brown"
            self.agent.is_target_color = self.agent.is_brown
            return True

        return False

    def is_chasing_done(self):
        return self.agent.reward is not None and self.agent.reward > 0

    def synthesize_is_inaccessible(self):
        # TODO maybe add the walls here, but...
        is_inaccessible = np.copy(self.agent.is_red)
        is_inaccessible = np.logical_and(is_inaccessible, np.logical_not(AgentConstants.frame_mask))
        return is_inaccessible

    def find_reachable_object(self):
        labels, label_num = measure.label(input=self.agent.is_color, background=False, return_num=True, connectivity=1)
        sizes = [(labels == label).sum() for label in range(1, label_num + 1)]

        lowest_idx = None
        for ii in np.argsort(sizes)[::-1]:
            label = ii + 1
            idx = np.argwhere(labels = label)
            idx_idx = idx.argmax(axis = 0)[1]
            lowest_idx = idx[idx_idx]

        is_stand

        target_label = np.argmax(sizes) + 1
        target_center = np.array(np.where(labels == target_label)).mean(axis=1).astype(np.int)
        target_size = sizes[target_label - 1]

        return target_center, target_size
