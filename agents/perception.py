import AgentConstants
import numpy as np
import warnings
from skimage import measure
import agentUtils
from scipy.ndimage.interpolation import shift


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
        # if self.agent.is_orange_memory.full():
        #     self.agent.is_orange_memory.get()
        if self.agent.is_yellow_memory.full():
            self.agent.is_yellow_memory.get()
        if self.agent.vector_memory.full():
            self.agent.vector_memory.get()

        self.agent.is_green = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "green")) < AgentConstants.green_tolerance
        # self.agent.is_brown = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
        #     "brown")) < AgentConstants.brown_tolerance
        self.agent.is_brown = abs(self.agent.obs_visual - AgentConstants.predefined_colors.get("brown")).max(
            axis=2) < AgentConstants.brown_tolerance
        self.agent.is_brown = self.agent.is_brown if agentUtils.is_color_significant(
            self.agent.is_brown, AgentConstants.size_limit) else AgentConstants.all_false
        self.agent.is_red = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "red")) < AgentConstants.red_tolerance
        self.agent.is_red = self.puff_red(delta=2)
        self.agent.is_gray = np.logical_and((self.agent.obs_visual[:, :, 0] == self.agent.obs_visual[:, :, 1]),
                                            (self.agent.obs_visual[:, :, 1] == self.agent.obs_visual[:, :, 2]))
        self.agent.is_blue = np.logical_and(np.logical_and(
                                abs(self.agent.obs_visual[:, :, 0] - AgentConstants.predefined_colors.get(
                                 "sky_blue")[0]) < AgentConstants.sky_blue_tolerance,
                                abs(self.agent.obs_visual[:, :, 1] - AgentConstants.predefined_colors.get(
                                 "sky_blue")[1]) < AgentConstants.sky_blue_tolerance),
                                abs(self.agent.obs_visual[:, :, 2] - AgentConstants.predefined_colors.get(
                                 "sky_blue")[2]) < AgentConstants.sky_blue_tolerance)
        # self.agent.is_orange = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
        #     "orange")) < AgentConstants.orange_tolerance
        self.agent.is_yellow = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "yellow")) < AgentConstants.yellow_tolerance
        self.synthesize_is_inaccessible()
        self.update_target()
        self.update_nearest_inaccessible_idx()

        self.agent.visual_h_memory.put(self.agent.obs_visual_h)
        self.agent.is_green_memory.put(self.agent.is_green)
        self.agent.is_brown_memory.put(self.agent.is_brown)
        self.agent.is_red_memory.put(self.agent.is_red)
        # self.agent.is_orange_memory.put(self.agent.is_orange)
        self.agent.is_yellow_memory.put(self.agent.is_yellow)
        self.agent.vector_memory.put(self.agent.obs_vector[0])

    def renew_target_from_panorama(self):

        if self.agent.pirouette_step_n == AgentConstants.pirouette_step_limit:

            green_memory = np.array(self.agent.is_green_memory.queue)[-AgentConstants.pirouette_step_limit:]
            if green_memory.any():
                self.agent.target_color = "green"
                best_direction = green_memory.sum(axis = (1, 2)).argmax()
                if 0 <= best_direction < 30:
                    if self.agent.search_direction == AgentConstants.left:
                        self.agent.search_direction = AgentConstants.left
                    else:
                        self.agent.search_direction = AgentConstants.right
                else:
                    if self.agent.search_direction == AgentConstants.left:
                        self.agent.search_direction = AgentConstants.right
                    else:
                        self.agent.search_direction = AgentConstants.left
            else:
                self.agent.safest_direction = np.random.choice(AgentConstants.pirouette_step_limit)
                # is_yellow = np.array(self.agent.is_yellow_memory.queue)[-AgentConstants.pirouette_step_limit:]
                # if is_yellow.any():
                #     self.agent.safest_direction = np.argmax(
                #         [np.logical_and(frame, AgentConstants.road_mask).sum() for frame in is_yellow])
                # else:
                #     warnings.warn("Nowhere to go, just move forward")
                #     self.agent.safest_direction = 0

    def renew_target(self):

        if self.agent.target_color == "green" and self.agent.is_brown.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "brown"
                return True

        if self.agent.target_color is None and self.agent.is_green.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_green)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "green"
                return True

        if self.agent.target_color is None and self.agent.is_brown.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "brown"
                return True

        return False

    def is_front_safe(self):
        if self.agent.nearest_inaccessible_idx is None:
            return True
        else:
            return (AgentConstants.resolution - self.agent.nearest_inaccessible_idx[0]) > \
                   AgentConstants.minimal_dist_to_in_accessible

    def is_static(self):
        return (self.agent.obs_vector == 0).all()

    def is_nearly_static(self):
        return (abs(self.agent.obs_vector) < 0.1).all()

    def is_found(self):
        return self.agent.reachable_target_idx is not None

    def is_chasing_done(self):
        return self.agent.reward is not None and self.agent.reward > 0

    def synthesize_is_inaccessible(self):
        # TODO maybe add the walls here, but...
        self.agent.is_inaccessible = np.logical_or(self.agent.is_gray, self.agent.is_red)
        self.agent.is_inaccessible = np.logical_or(self.agent.is_inaccessible, self.agent.is_blue)
        self.agent.is_inaccessible_masked = np.logical_and(self.agent.is_inaccessible, np.logical_not(AgentConstants.frame_mask))

    def find_reachable_target(self, is_color):
        if is_color.any():
            labels, label_num = measure.label(input=is_color,
                                              background=False,
                                              return_num=True,
                                              connectivity=1)
            sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
            for ii in np.argsort(sizes)[::-1]:
                label = ii + 1
                idx = np.argwhere(labels == label)
                idx_idx = idx.argmax(axis=0)[0]
                lowest_idx = idx[idx_idx]
                return lowest_idx, sizes[ii]
            #     if not self.agent.is_inaccessible[lowest_idx[0]: lowest_idx[0] + 3, lowest_idx[1]].any():
            #         return lowest_idx, sizes[ii]
            #
            # return None, None
        else:
            return None, None

    def update_target(self):

        if self.agent.is_brown.any():
            self.agent.target_color = "brown"
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            return
        elif self.agent.target_color == "green" and self.agent.is_green.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_green)
            return
        else:
            self.agent.reachable_target_idx = None
            self.agent.reachable_target_size = None

    def update_nearest_inaccessible_idx(self):
        idx = np.argwhere(np.logical_and(AgentConstants.road_mask, self.agent.is_red))
        if 0 == len(idx):
            self.agent.nearest_inaccessible_idx = None
        else:
            self.agent.nearest_inaccessible_idx = idx[idx[:, 0].argmax()]

    def puff_red(self, delta):
        new_is_red = self.agent.is_red.copy()
        for delt in np.arange(1, delta + 1):
            new_is_red = np.logical_or(new_is_red, shift(self.agent.is_red, (-delt, 0), cval=False))
            new_is_red = np.logical_or(new_is_red, shift(self.agent.is_red, (delt, 0), cval=False))
            new_is_red = np.logical_or(new_is_red, shift(self.agent.is_red, (0, -delt), cval=False))
            new_is_red = np.logical_or(new_is_red, shift(self.agent.is_red, (0, delt), cval=False))
        new_is_red = np.logical_and(np.logical_and(new_is_red,
                                                   np.logical_not(self.agent.is_green)),
                                                   np.logical_not(self.agent.is_brown))
        return new_is_red

    def reset(self):
        pass
