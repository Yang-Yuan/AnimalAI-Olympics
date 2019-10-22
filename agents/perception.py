import AgentConstants


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
            return self.agent.is_brown
