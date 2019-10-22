import AgentConstants
import numpy as np
import warnings


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
        if self.agent.target_color == "green":
            return self.agent.is_green.any()
        elif self.agent.target_color == "brown":
            return self.agent.is_brown.any()

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
