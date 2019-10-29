import AgentConstants
import warnings
import numpy as np


class Strategy(object):

    def __init__(self, agent):
        self.agent = agent
        self.static_step_n = None

    def run_strategy(self):

        # if done
        if self.agent.done:
            self.agent.currentAction = AgentConstants.taxi
            return

        while True:

            # if the agent is static
            if self.agent.actionStateMachine.is_static:
                if 0 == self.agent.pirouette_step_n:
                    self.agent.actionStateMachine.search()
                    break
                else:
                    warnings.warn("undefined branch!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.agent.actionStateMachine.reset()
                    break

            # if the agent is searching
            elif self.agent.actionStateMachine.is_searching:
                if self.agent.perception.is_found():
                    self.agent.actionStateMachine.chase()
                    break
                else:
                    if self.agent.pirouette_step_n < AgentConstants.pirouette_step_limit:
                        self.agent.actionStateMachine.hold()
                        break
                    else:
                        if self.agent.safest_direction is not None:
                            self.agent.actionStateMachine.rotate_to_direction()
                            break
                        else:
                            self.agent.actionStateMachine.search()
                            break

            # if the agent is rotating_to_direction
            elif self.agent.actionStateMachine.is_rotating_to_direction:
                if self.agent.perception.is_found():
                    self.agent.actionStateMachine.chase()
                    break
                elif self.agent.safest_direction == 0:
                    if self.agent.perception.is_front_safe():
                        self.agent.actionStateMachine.roam()
                        break
                    else:
                        self.agent.actionStateMachine.search()
                        break
                else:
                    self.agent.actionStateMachine.hold()
                    break

            # if the agent is roaming
            elif self.agent.actionStateMachine.is_roaming:
                if self.agent.perception.is_found():
                    self.agent.actionStateMachine.chase()
                    break
                elif self.agent.perception.is_front_safe() and not self.agent.perception.is_nearly_static():
                    self.agent.actionStateMachine.hold()
                    break
                else:
                    self.agent.actionStateMachine.decelerate()
                    break

            # if the agent is chasing
            elif self.agent.actionStateMachine.is_chasing:
                if self.agent.perception.is_chasing_done():
                    if self.agent.perception.is_found():
                        self.agent.actionStateMachine.chase()
                        break
                    else:
                        self.agent.actionStateMachine.decelerate()
                        break
                elif self.agent.chase_failed:
                    self.agent.actionStateMachine.decelerate()
                    break
                else:
                    if self.agent.not_seeing_target_step_n < AgentConstants.not_seeing_target_step_limit:
                        self.agent.actionStateMachine.hold()
                        break
                    else:
                        self.agent.actionStateMachine.decelerate()
                        break

            # if the agent is decelerating
            elif self.agent.actionStateMachine.is_decelerating:
                if self.agent.perception.is_found():
                    self.agent.actionStateMachine.chase()
                    break
                elif self.agent.perception.is_static():
                    self.agent.actionStateMachine.search()
                    break
                else:
                    self.agent.actionStateMachine.hold()
                    break

            # if the agent is in an unknown state
            else:
                warnings.warn("An unknown state: {}".format(self.agent.actionStateMachine.current_state))
                self.agent.actionStateMachine.reset()
                break

        self.deadlock_breaker()

    def deadlock_breaker(self):

        if self.agent.actionStateMachine.is_chasing:

            # if self.agent.currentAction == AgentConstants.left or self.agent.currentAction == AgentConstants.right \
            #         or self.agent.currentAction == AgentConstants.taxi:
            #     self.static_step_n += 1
            # else:
            #     self.static_step_n = 0

            if self.agent.perception.is_nearly_static():
                self.static_step_n += 1
            else:
                self.static_step_n = 0

            if self.static_step_n > AgentConstants.deadlock_step_limit:
                self.agent.safest_direction = np.random.choice(AgentConstants.directions_for_deadlock)
                self.agent.actionStateMachine.rotate_to_direction()
                self.static_step_n = 0
        else:
            self.static_step_n = 0

    def reset(self):
        self.static_step_n = 0
