import AgentConstants
import warnings
import numpy as np


class Strategy(object):

    def __init__(self, agent):
        self.agent = agent

    def run_strategy(self):
        '''
        Using the primitive and high-level perception,
        this method will set the agent's action which will be return to the env.
        :return:
        '''

        # if the env signals the end of the current test, do nothing (taxi is the action of [0, 0])
        if self.agent.done:
            self.agent.currentAction = AgentConstants.taxi
            return

        # this big while loop is the main part of my strategy.
        # It is using a state machine defined in ActionStateMachine.py to implement the strategy.
        # Basically, it determines which state the state machine is in, and then decides which action
        # it should take to move to the next state. By action, here I mean the actions of the state machine
        # (the edges of state machine graph if you like). It is different from the actions of the agent, which are
        # to move forward or backward and turn left or right. The former is high-level actions, while the later is
        # low-level actions.
        while True:

            # if the state is static
            if self.agent.action_state_machine.is_static:
                if 0 == self.agent.pirouette_step_n:
                    self.agent.action_state_machine.search()
                    break
                else:
                    warnings.warn("undefined branch!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.agent.action_state_machine.reset()
                    break

            # if the state is searching
            elif self.agent.action_state_machine.is_searching:
                if self.agent.perception.is_found():
                    self.agent.action_state_machine.chase()
                    break
                else:
                    if self.agent.pirouette_step_n < AgentConstants.pirouette_step_limit:
                        self.agent.action_state_machine.hold()
                        break
                    else:
                        if self.agent.exploratory_direction is not None:
                            self.agent.action_state_machine.rotate_to_direction()
                            break
                        else:
                            self.agent.action_state_machine.search()
                            break

            # if the state is rotating_to_direction
            elif self.agent.action_state_machine.is_rotating_to_direction:
                if self.agent.perception.is_found():
                    self.agent.action_state_machine.chase()
                    break
                elif self.agent.exploratory_direction == 0:
                    if self.agent.perception.is_front_safe():
                        self.agent.action_state_machine.roam()
                        break
                    else:
                        self.agent.action_state_machine.search()
                        break
                else:
                    self.agent.action_state_machine.hold()
                    break

            # if the state is roaming
            elif self.agent.action_state_machine.is_roaming:
                if self.agent.perception.is_found():
                    self.agent.action_state_machine.chase()
                    break
                elif self.agent.perception.is_front_safe() and not self.agent.perception.is_nearly_static():
                    self.agent.action_state_machine.hold()
                    break
                else:
                    self.agent.action_state_machine.decelerate()
                    break

            # if the state is chasing
            elif self.agent.action_state_machine.is_chasing:
                if self.agent.perception.is_chasing_done():
                    if self.agent.perception.is_found():
                        self.agent.action_state_machine.chase()
                        break
                    else:
                        self.agent.action_state_machine.decelerate()
                        break
                elif self.agent.chase_failed:
                    self.agent.action_state_machine.decelerate()
                    break
                else:
                    if self.agent.not_seeing_target_step_n < AgentConstants.not_seeing_target_step_limit:
                        self.agent.action_state_machine.hold()
                        break
                    else:
                        self.agent.action_state_machine.decelerate()
                        break

            # if the state is decelerating
            elif self.agent.action_state_machine.is_decelerating:
                if self.agent.perception.is_found():
                    self.agent.action_state_machine.chase()
                    break
                elif self.agent.perception.is_static():
                    self.agent.action_state_machine.search()
                    break
                else:
                    self.agent.action_state_machine.hold()
                    break

            # if the state is in an unknown state (this won't be possible)
            else:
                warnings.warn("An unknown state: {}".format(self.agent.action_state_machine.current_state))
                self.agent.action_state_machine.reset()
                break

        self.deadlock_breaker()

    def deadlock_breaker(self):

        if self.agent.action_state_machine.is_chasing:

            if self.agent.perception.is_nearly_static():
                self.agent.static_step_n += 1
            else:
                self.agent.static_step_n = 0

            if self.agent.static_step_n > AgentConstants.deadlock_step_limit:
                self.agent.exploratory_direction = np.random.choice(AgentConstants.directions_for_deadlock)
                self.agent.action_state_machine.rotate_to_direction()
                self.agent.static_step_n = 0
        else:
            self.agent.static_step_n = 0

    def reset(self):
        pass
