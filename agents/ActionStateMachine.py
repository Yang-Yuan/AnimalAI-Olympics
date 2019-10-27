from statemachine import StateMachine, State
import AgentConstants
import numpy as np
import warnings
import sys


class ActionStateMachine(StateMachine):
    # ************************** states ***************************
    static = State("static", initial=True)
    # 1 agent not moving, initial state
    pirouetting = State("pirouetting")
    # 2 rotating in situ to observe
    rotating_to_direction = State("rotating_to_direction")
    # 3 not finding any good balls
    roaming = State("roaming")
    # 4 moving forward for a distance if possible
    searching = State("searching")
    # 5 targeting a specific color
    chasing = State("chasing")
    # 6 avoiding the bad zones
    decelerating = State("decelerating")
    # 7 let the speed decrease spontaneously
    # ************************** states end ***************************

    # ************************** actions ***************************
    hold = static.to.itself() | \
           pirouetting.to.itself() | \
           rotating_to_direction.to.itself() | \
           roaming.to.itself() | \
           searching.to.itself() | \
           chasing.to.itself() | \
           decelerating.to.itself()

    pirouette = static.to(pirouetting) | \
                decelerating.to(pirouetting)

    rotate_to_direction = pirouetting.to(rotating_to_direction)

    roam = pirouetting.to(roaming) | \
           rotating_to_direction.to(roaming)

    search = pirouetting.to(searching) | \
             chasing.to(searching)

    chase = searching.to(chasing) | pirouetting.to(chasing) | rotating_to_direction.to(chasing) | roaming.to(chasing) \
            | chasing.to(chasing)

    decelerate = chasing.to(decelerating) | \
                 roaming.to(decelerating)

    reset = static.to.itself() | \
            pirouetting.to(static) | \
            rotating_to_direction.to(static) | \
            roaming.to(static) | \
            searching.to(static) | \
            chasing.to(static) | \
            decelerating.to(static)

    # ************************** actions end***************************

    def __init__(self, agent):
        self.agent = agent
        super(ActionStateMachine, self).__init__()

    # ************************** callbacks for pirouette ***************************

    def on_pirouette(self):
        print("on_pirouette~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.pirouette_step_n = 0

    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        self.agent.currentAction = AgentConstants.left
        self.agent.pirouette_step_n += 1
        self.agent.perception.renew_target_from_panorama()

    # ************************** callbacks for pirouette end***************************

    # ************************** callbacks for roam ***************************
    def on_roam(self):
        print("on_roam~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def on_enter_roaming(self):
        print("on_roaming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.currentAction = AgentConstants.forward

    # ************************** callbacks for roam end***************************

    # ************************** callbacks for search ***************************
    def on_search(self):
        print("on_target~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.pirouette_step_n = 0

    def on_enter_searching(self):
        print("on_enter_searching: {}".format(self.agent.pirouette_step_n))
        self.agent.currentAction = AgentConstants.left
        self.agent.pirouette_step_n += 1
        self.agent.perception.renew_target_from_panorama()

    # ************************** callbacks for search ends***************************

    # ************************** callbacks for rotate_to_direction ***************************
    def on_rotate_to_direction(self):
        print("on_rotate_to_direction~~~~~~~~~~~~~~~~~~~~~~")
        if AgentConstants.pirouette_step_limit / 2 <= self.agent.safest_direction \
                < AgentConstants.pirouette_step_limit:
            self.agent.safest_direction -= 60

    def on_enter_rotating_to_direction(self):
        print("on_enter_rotating_to_direction~~~~~~~~~~~~~~~~~")
        if self.agent.safest_direction > 0:
            self.agent.currentAction = AgentConstants.left
            self.agent.safest_direction -= 1
        else:
            self.agent.currentAction = AgentConstants.right
            self.agent.safest_direction += 1

    # ************************** callbacks for rotate_to_direction end***************************

    # ************************** callbacks for decelerate ***************************
    def on_enter_decelerating(self):
        print("on_enter_decelerating~~~~~~~~~~~~~~~~~~~~~")
        self.agent.currentAction = AgentConstants.taxi

    # ************************** callbacks for decelerate end***************************

    # ************************** callbacks for chase ***************************
    def on_chase(self):
        print("on_chase~~~~~~~~~~~~~~~~~")
        self.agent.not_seeing_target_step_n = 0
        self.agent.chase_failed = False
        self.agent.chaser.newest_path = None

    def on_enter_chasing(self):
        print("on_enter_chasing~~~~~~~~~~~~~~~~~~~~~~~~")
        if self.agent.reachable_target_idx is None:
            self.agent.not_seeing_target_step_n += 1
            self.agent.chaser.chase_in_dark()
        else:
            self.agent.not_seeing_target_step_n = 0
            self.agent.chaser.chase()
    # ************************** callbacks for chase ends ***************************
