from statemachine import StateMachine, State
import AgentConstants
from skimage import measure
import numpy as np


class ActionStateMachine(StateMachine):
    # ************************** states ***************************
    static = State("static", initial=True)
    # agent not moving
    pirouetting = State("pirouetting")
    # rotating in situ to observe
    roaming = State("roaming")
    # not finding any good balls
    targeting = State("targeting")
    # targeting a specific color
    accelerating = State("accelerating")
    # avoiding the bad zones
    stopping = State("stopping")
    # let the speed decrease spontaneously
    # ************************** states end ***************************

    # ************************** actions ***************************
    hold = static.to.itself() | pirouetting.to.itself() | targeting.to.itself() | accelerating.to.itself()
    pirouette = static.to(pirouetting)
    roam = static.to(roaming)
    target = pirouetting.to(targeting) | accelerating.to(targeting)
    accelerate = targeting.to(accelerating)
    slowdown = accelerating.to(stopping)
    stop = stopping.to(static)
    analyze_panorama = pirouetting.to(static)
    reset = static.to.itself() | pirouetting.to(static) | targeting.to(static) | accelerating.to(static) | stopping.to(
        static)

    # ************************** actions end***************************

    def __init__(self, agent):
        self.agent = agent
        super(ActionStateMachine, self).__init__()

    # ************************** action callbacks ***************************
    def on_pirouette(self):
        print("on_pirouette~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.pirouette_step_n = 0
        self.agent.visual_memory.fill(-1)
        self.agent.target_color = None
        self.agent.spacious_direction = None

    def on_roam(self):
        print("on_roam~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def on_target(self):
        print("on_target~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def on_stop(self):
        print("on_stop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.currentAction = [0, 0]

    def on_analyze_panorama(self):
        print("on_analyzePanorama~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.is_green = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "green")) < AgentConstants.color_diff_limit
        self.agent.is_brown = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "brown")) < AgentConstants.color_diff_limit
        self.agent.is_red = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "red")) < AgentConstants.color_diff_limit
        self.agent.is_orange = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "orange")) < AgentConstants.color_diff_limit
        self.agent.is_yellow = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "yellow")) < AgentConstants.color_diff_limit
        if self.agent.is_brown.any():
            self.agent.target_color = "brown"
            return
        elif self.agent.is_green.any():
            self.agent.target_color = "green"
            return
        elif self.agent.is_yellow.any():
            self.agent.spacious_direction = np.argmax(
                [(frame & AgentConstants.road_mask).sum() for frame in self.agent.is_yellow])

    # ************************** action callbacks end***************************

    # ************************** state callbacks ***************************
    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        self.agent.visual_memory[self.agent.pirouette_step_n] = self.agent.obs_visual_h
        self.agent.currentAction = [0, 1]
        self.agent.pirouette_step_n += 1

    def on_enter_roaming(self):
        print("on_enter_roaming~~~~~~~~~~~~~~~~~")

    # ************************** state callbacks ends***************************
