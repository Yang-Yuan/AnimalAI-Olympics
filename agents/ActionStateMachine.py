from statemachine import StateMachine, State
import AgentConstants
from skimage import measure
import numpy as np


class ActionStateMachine(StateMachine):
    # ************************** states ***************************
    static = State("static", initial=True)
    # agent not moving
    pirouetting = State("pirouetting")
    # rotating in situ
    targeting = State("pursuing")
    # targeting a specific color
    accelerating = State("avoiding")
    # avoiding the bad zones
    stopping = State("stopping")
    # let the speed decrease spontaneously
    # ************************** states end ***************************

    # ************************** actions ***************************
    hold = static.to.itself() | pirouetting.to.itself() | targeting.to.itself() | accelerating.to.itself()
    pirouette = static.to(pirouetting)
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
        self.agent.green_labels = None
        self.agent.green_sizes = None
        self.agent.brown_labels = None
        self.agent.brown_sizes = None
        self.agent.red_labels = None
        self.agent.red_sizes = None
        self.agent.orange_labels = None
        self.agent.orange_sizes = None

    def on_stop(self):
        print("on_stop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.currentAction = [0, 0]

    def on_analyze_panorama(self):
        print("on_analyzePanorama~~~~~~~~~~~~~~~~~~~~~~~~")
        is_green = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "green")) < AgentConstants.color_diff_limit
        is_brown = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "brown")) < AgentConstants.color_diff_limit
        is_red = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "red")) < AgentConstants.color_diff_limit
        is_orange = abs(self.agent.visual_memory - AgentConstants.predefined_colors_h.get(
            "orange")) < AgentConstants.color_diff_limit

        if is_green.any():
            self.agent.green_labels, green_label_num = measure.label(input=is_green, background=False, return_num=True)
            self.agent.green_sizes = [(self.agent.green_labels == label).sum() for label in range(1, green_label_num + 1)]

        if is_brown.any():
            self.agent.brown_labels, brown_label_num = measure.label(input=is_brown, background=False, return_num=True)
            self.agent.brown_sizes = [(self.agent.brown_labels == label).sum() for label in range(1, brown_label_num + 1)]

        if is_red.any():
            self.agent.red_labels, red_label_num = measure.label(input=is_red, background=False, return_num=True)
            self.agent.red_sizes = [(self.agent.red_labels == label).sum() for label in range(1, red_label_num + 1)]

        if is_orange.any():
            self.agent.orange_labels, orange_label_num = measure.label(input=is_orange, background=False, return_num=True)
            self.agent.orange_sizes = [(self.agent.orange_labels == label).sum() for label in range(1, orange_label_num + 1)]

        if is_brown.any():
            target_label = np.argmax(self.agent.brown_sizes) + 1
            center_of_target = np.array(np.where(self.agent.brown_labels == target_label)).mean(axis=1)
            target_size = self.agent.brown_sizes[target_label - 1]
    # ************************** action callbacks end***************************

    # ************************** state callbacks ***************************
    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        self.agent.visual_memory[self.agent.pirouette_step_n] = self.agent.obs_visual_h
        self.agent.currentAction = [0, 1]
        self.agent.pirouette_step_n += 1

    # ************************** state callbacks ends***************************
