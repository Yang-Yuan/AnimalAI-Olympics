from statemachine import StateMachine, State
import AgentConstants


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

    # ************************** actions ***************************
    hold = static.to.itself() | pirouetting.to.itself() | targeting.to.itself() | accelerating.to.itself()
    pirouette = static.to(pirouetting)
    target = pirouetting.to(targeting) | accelerating.to(targeting)
    accelerate = targeting.to(accelerating)
    slowdown = accelerating.to(stopping)
    stop = stopping.to(static) | pirouetting.to(static)
    reset = static.to.itself() | pirouetting.to(static) | targeting.to(static) | accelerating.to(static) | stopping.to(
        static)

    def __init__(self, agent):
        self.agent = agent
        super(ActionStateMachine, self).__init__()

    def on_pirouette(self):
        print("on_pirouette~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.pirouette_step_n = 0
        self.agent.currentAction = [0, 1]

    def on_stop(self):
        print("on_stop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.agent.currentAction = [0, 0]

    def on_enter_pirouetting(self):
        print("on_enter_pirouetting: {}".format(self.agent.pirouette_step_n))
        if self.agent.pirouette_step_n < AgentConstants.pirouette_step_limit:
            self.agent.pirouette_step_n += 1
