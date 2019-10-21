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
    rotating_to_direction = State("rotating_to_direction")
    # not finding any good balls
    roaming = State("roaming")
    # moving forward for a distance if possible
    targeting = State("targeting")
    # targeting a specific color
    accelerating = State("accelerating")
    # avoiding the bad zones
    decelerating = State("decelerating")
    # let the speed decrease spontaneously
    # ************************** states end ***************************

    # ************************** actions ***************************
    hold = static.to.itself() | pirouetting.to.itself() | targeting.to.itself() | \
           accelerating.to.itself() | rotating_to_direction.to.itself() | roaming.to.itself()
    pirouette = static.to(pirouetting)
    rotate_to_direction = static.to(rotating_to_direction)
    roam = rotating_to_direction.to(roaming)
    target = pirouetting.to(targeting) | accelerating.to(targeting)
    accelerate = targeting.to(accelerating)
    decelerate = accelerating.to(decelerating) | roaming.to(decelerating)
    stop = decelerating.to(static)
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
        self.agent.roaming_step_n = np.random.randint(low = 1, high = AgentConstants.roam_step_limit)

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

    def on_rotate_to_direction(self):
        print("on_rotate_to_direction~~~~~~~~~~~~~~~~~~~~~~")
        if AgentConstants.pirouette_step_limit / 2 <= self.agent.spacious_direction < AgentConstants.pirouette_step_limit:
            self.agent.spacious_direction -= 60

    # ************************** action callbacks end***************************

    # ************************** state callbacks ***************************
    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        self.agent.visual_memory[self.agent.pirouette_step_n] = self.agent.obs_visual_h
        self.agent.currentAction = AgentConstants.left
        self.agent.pirouette_step_n += 1

    def on_enter_rotating_to_direction(self):
        print("on_enter_roaming~~~~~~~~~~~~~~~~~")
        if self.agent.spacious_direction > 0:
            self.agent.currentAction = AgentConstants.left
            self.agent.spacious_direction -= 1
        else:
            self.agent.currentAction = AgentConstants.right
            self.agent.spacious_direction += 1

    def on_roaming(self):
        self.agent.currentAction = AgentConstants.forward
        self.agent.roaming_step_n -= 1


    # ************************** state callbacks ends***************************
