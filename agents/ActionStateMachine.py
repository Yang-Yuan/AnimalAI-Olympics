from statemachine import StateMachine, State
import AgentConstants


class ActionStateMachine(StateMachine):
    # ************************** states ***************************
    static = State("static", initial=True)
    # 1 agent not moving, initial state
    searching = State("searching")
    # 2 searching for a specific color
    rotating_to_direction = State("rotating_to_direction")
    # 3 not finding any good balls
    roaming = State("roaming")
    # 4 moving forward for a distance if possible
    chasing = State("chasing")
    # 5 avoiding the bad zones
    decelerating = State("decelerating")
    # 6 let the speed decrease spontaneously
    # ************************** states end ***************************

    # ************************** actions ***************************
    hold = static.to.itself() | \
           rotating_to_direction.to.itself() | \
           roaming.to.itself() | \
           searching.to.itself() | \
           chasing.to.itself() | \
           decelerating.to.itself()

    rotate_to_direction = searching.to(rotating_to_direction) | \
                          chasing.to(rotating_to_direction)

    roam = rotating_to_direction.to(roaming) | \
           searching.to(roaming)

    search = static.to(searching) | \
             searching.to(searching) | \
             decelerating.to(searching) | \
             rotating_to_direction.to(searching)

    chase = searching.to(chasing) | \
            rotating_to_direction.to(chasing) | \
            roaming.to(chasing) | \
            chasing.to(chasing) | \
            decelerating.to(chasing)

    decelerate = chasing.to(decelerating) | \
                 roaming.to(decelerating)

    reset = static.to.itself() | \
            rotating_to_direction.to(static) | \
            roaming.to(static) | \
            searching.to(static) | \
            chasing.to(static) | \
            decelerating.to(static)

    # ************************** actions end***************************

    def __init__(self, agent):
        self.agent = agent
        super(ActionStateMachine, self).__init__()

    # ************************** callbacks for static ***************************
    def on_enter_static(self):
        self.agent.search_direction = AgentConstants.left
    # ************************** callbacks for static end***************************

    # ************************** callbacks for roam ***************************
    def on_roam(self):
        # print("on_roam~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        pass

    def on_enter_roaming(self):
        # print("on_roaming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.current_action = AgentConstants.forward

    # ************************** callbacks for roam end***************************

    # ************************** callbacks for search ***************************
    def on_search(self):
        # print("on_target~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.pirouette_step_n = 0

    def on_enter_searching(self):
        # print("on_enter_searching: {}".format(self.agent.pirouette_step_n))
        self.agent.current_action = self.agent.search_direction
        self.agent.pirouette_step_n += 1
        self.agent.perception.renew_target_from_panorama()

    # ************************** callbacks for search ends***************************

    # ************************** callbacks for rotate_to_direction ***************************
    def on_rotate_to_direction(self):
        # print("on_rotate_to_direction~~~~~~~~~~~~~~~~~~~~~~")
        if AgentConstants.pirouette_step_limit / 2 <= self.agent.exploratory_direction \
                < AgentConstants.pirouette_step_limit:
            self.agent.exploratory_direction -= 60

    def on_enter_rotating_to_direction(self):
        # print("on_enter_rotating_to_direction~~~~~~~~~~~~~~~~~")
        if self.agent.exploratory_direction > 0:
            self.agent.current_action = AgentConstants.left
            self.agent.exploratory_direction -= 1
        else:
            self.agent.current_action = AgentConstants.right
            self.agent.exploratory_direction += 1

    # ************************** callbacks for rotate_to_direction end***************************

    # ************************** callbacks for decelerate ***************************
    def on_decelerate(self):
        self.agent.target_color = "brown"

    def on_enter_decelerating(self):
        # print("on_enter_decelerating~~~~~~~~~~~~~~~~~~~~~")
        self.agent.current_action = AgentConstants.taxi

    # ************************** callbacks for decelerate end***************************

    # ************************** callbacks for chase ***************************
    def on_chase(self):
        # print("on_chase~~~~~~~~~~~~~~~~~")
        self.agent.not_seeing_target_step_n = 0
        self.agent.chase_failed = False
        self.agent.chaser.newest_path = None
        self.agent.chaser.newest_end = None

    def on_enter_chasing(self):
        # print("on_enter_chasing~~~~~~~~~~~~~~~~~~~~~~~~")
        if self.agent.reachable_target_idx is None:
            self.agent.not_seeing_target_step_n += 1
            self.agent.chaser.chase_in_dark()
        else:
            self.agent.not_seeing_target_step_n = 0
            self.agent.chaser.chase()
    # ************************** callbacks for chase ends ***************************
