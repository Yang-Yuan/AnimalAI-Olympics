from statemachine import StateMachine, State
import AgentConstants
import numpy as np
import warnings


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
    searching = State("searching")
    # targeting a specific color
    chasing = State("chasing")
    # avoiding the bad zones
    decelerating = State("decelerating")
    # let the speed decrease spontaneously
# ************************** states end ***************************

# ************************** actions ***************************
    hold = static.to.itself() | \
           pirouetting.to.itself() | \
           searching.to.itself() | \
           chasing.to.itself() | \
           rotating_to_direction.to.itself() | \
           roaming.to.itself() | \
           decelerating.to.itself()

    pirouette = static.to(pirouetting) | \
                decelerating.to(pirouetting)

    rotate_to_direction = pirouetting.to(rotating_to_direction)

    roam = pirouetting.to(roaming) | \
           rotating_to_direction.to(roaming)

    search = pirouetting.to(searching) | \
             chasing.to(searching)

    chase = searching.to(chasing)

    decelerate = chasing.to(decelerating) | \
                 roaming.to(decelerating)

    reset = static.to.itself() | \
            pirouetting.to(static) | \
            searching.to(static) | \
            chasing.to(static) | \
            decelerating.to(static) | \
            roaming.to(static) | \
            rotating_to_direction.to(static)
# ************************** actions end***************************

    def __init__(self, agent):
        self.agent = agent
        super(ActionStateMachine, self).__init__()

# ************************** callbacks for pirouette ***************************

    def on_pirouette(self):
        print("on_pirouette~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.target_color = None
        self.agent.safest_direction = None

    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        self.agent.currentAction = AgentConstants.left
        self.agent.pirouette_step_n += 1

    def on_exit_pirouetting(self):
        print("exit_pirouetting~~~~~~~~~~~~~~~~~~")
        if self.agent.pirouette_step_n == AgentConstants.pirouette_step_limit:

            if np.array(self.agent.is_brown_memory.queue)[-AgentConstants.pirouette_step_limit:].any():
                self.agent.target_color = "brown"
                return
            elif np.array(self.agent.is_green_memory.queue)[-AgentConstants.pirouette_step_limit:].any():
                self.agent.target_color = "green"
                return
            else:
                is_yellow = np.array(self.agent.is_yellow_memory.queue)[-AgentConstants.pirouette_step_limit:]
                if is_yellow.any():
                    self.agent.safest_direction = np.argmax(
                        [(frame & AgentConstants.road_mask).sum() for frame in self.agent.is_yellow])
                else:
                    warnings.warn("Nowhere to go, just move forward")
                    self.agent.safest_direction = 0
# ************************** callbacks for pirouette end***************************

# ************************** callbacks for roam ***************************
    def on_roam(self):
        print("on_roam~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.roaming_step_n = np.random.randint(low = 1, high = AgentConstants.roam_step_limit)

    def on_roaming(self):
        self.agent.currentAction = AgentConstants.forward
        self.agent.roaming_step_n -= 1
# ************************** callbacks for roam end***************************

# ************************** callbacks for search ***************************
    def on_search(self):
        print("on_target~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# ************************** callbacks for search ends***************************


# ************************** callbacks for rotate_to_direction ***************************
    def on_rotate_to_direction(self):
        print("on_rotate_to_direction~~~~~~~~~~~~~~~~~~~~~~")
        if AgentConstants.pirouette_step_limit / 2 <= self.agent.spacious_direction \
                < AgentConstants.pirouette_step_limit:
            self.agent.spacious_direction -= 60

    def on_enter_rotating_to_direction(self):
        print("on_enter_roaming~~~~~~~~~~~~~~~~~")
        if self.agent.spacious_direction > 0:
            self.agent.currentAction = AgentConstants.left
            self.agent.spacious_direction -= 1
        else:
            self.agent.currentAction = AgentConstants.right
            self.agent.spacious_direction += 1
# ************************** callbacks for rotate_to_direction end***************************

# ************************** callbacks for decelerate ***************************
    def on_enter_decelerating(self):
        self.agent.currentAction = AgentConstants.taxi
# ************************** callbacks for decelerate end***************************

# ************************** callbacks for chase ***************************
    def on_chase(self):
        pass

    def on_chasing(self):
        pass
# ************************** callbacks for chase ends ***************************
