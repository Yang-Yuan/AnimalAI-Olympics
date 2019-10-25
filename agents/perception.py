import AgentConstants
import numpy as np
import warnings
import sys
from skimage import measure
import agentUtils


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
        self.agent.is_brown = abs(self.agent.obs_visual - AgentConstants.predefined_colors.get("brown")).max(axis = 2) < AgentConstants.brown_tolerance
        print(np.argwhere(self.agent.is_brown))
        self.agent.is_brown = self.agent.is_brown if agentUtils.is_color_significant(
            self.agent.is_brown, AgentConstants.size_limit) else AgentConstants.all_false
        self.agent.is_red = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "red")) < AgentConstants.red_tolerance
        # self.agent.is_orange = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
        #     "orange")) < AgentConstants.orange_tolerance
        self.agent.is_yellow = abs(self.agent.obs_visual_h - AgentConstants.predefined_colors_h.get(
            "yellow")) < AgentConstants.yellow_tolerance
        self.agent.is_inaccessible = self.synthesize_is_inaccessible()
        self.update_target()

        self.agent.visual_h_memory.put(self.agent.obs_visual_h)
        self.agent.is_green_memory.put(self.agent.is_green)
        self.agent.is_brown_memory.put(self.agent.is_brown)
        self.agent.is_red_memory.put(self.agent.is_red)
        # self.agent.is_orange_memory.put(self.agent.is_orange)
        self.agent.is_yellow_memory.put(self.agent.is_yellow)
        self.agent.vector_memory.put(self.agent.obs_vector)

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
                        [np.logical_and(frame, AgentConstants.road_mask).sum() for frame in is_yellow])
                else:
                    warnings.warn("Nowhere to go, just move forward")
                    self.agent.safest_direction = 0
            print("target_color renewed: {} and safest_direction renewed: {}".format(self.agent.target_color,
                                                                                     self.agent.safest_direction))

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
        return (self.agent.is_red & AgentConstants.road_mask).sum() < AgentConstants.red_pixel_on_road_limit

    def is_static(self):
        return (self.agent.obs_vector == 0).all()

    def is_found(self):
        return self.agent.reachable_target_idx is not None

    def is_chasing_done(self):
        return self.agent.reward is not None and self.agent.reward > 0

    def synthesize_is_inaccessible(self):
        # TODO maybe add the walls here, but...
        is_inaccessible = np.copy(self.agent.is_red)
        is_inaccessible = np.logical_and(is_inaccessible, np.logical_not(AgentConstants.frame_mask))
        return is_inaccessible

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

                if not self.agent.is_inaccessible[lowest_idx[0]: lowest_idx[0] + 6, lowest_idx[1]].any():
                    return lowest_idx, sizes[ii]
        else:
            return None, None

    def update_target(self):
        if self.agent.target_color is None:
            return
        else:
            if self.agent.target_color == "green":
                self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                    self.agent.is_green)
                return
            elif self.agent.target_color == "brown":
                self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                    self.agent.is_brown)
                return
            else:
                warnings.warn("unkown corlor!!!!!!!!!!!!!!!!!!")
                sys.exit(1)

    def reset(self):
        pass
