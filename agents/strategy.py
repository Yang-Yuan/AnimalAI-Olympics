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
        # All the definition of states and actions of the state machine can be found in ActionStateMachine.py.
        # All the actions of the agent will be set in the callback functions for actions and states of the state machine.
        while True:

            # if the state is static (initial state of the state machine),
            # start searching for target by rotating
            if self.agent.action_state_machine.is_static:
                self.agent.action_state_machine.search()
                break

            # if the state is searching for the target
            elif self.agent.action_state_machine.is_searching:
                # if found the target, then chase it.
                if self.agent.perception.is_target_found():
                    self.agent.action_state_machine.chase()
                    break
                # if not found, then make decision according to how long it has been rotating
                else:
                    # if not long enough, keep rotating (searching)
                    if self.agent.pirouette_step_n < AgentConstants.pirouette_step_limit:
                        self.agent.action_state_machine.hold()
                        break
                    # if long enough, make decision according to whether it came across some
                    # new targets (the color of which are not the color it was looking for) while rotating
                    else:
                        # if no new targets, explore a random direction by first rotating to that direction
                        if self.agent.exploratory_direction is not None:
                            self.agent.action_state_machine.rotate_to_direction()
                            break
                        # if found new targets, then search for it
                        else:
                            self.agent.action_state_machine.search()
                            break

            # if the state is rotating_to_direction
            elif self.agent.action_state_machine.is_rotating_to_direction:
                # if some target is found while rotating to a certain direction (due to moving targets),
                # then chase it
                if self.agent.perception.is_target_found():
                    self.agent.action_state_machine.chase()
                    break
                # exploratory_direction is decreased by 1 every step. If it is zero,
                # the agent should be facing the exploratory direction
                elif self.agent.exploratory_direction == 0:
                    # if it is safe to go in this direction, then go (roaming)
                    if self.agent.perception.is_front_safe():
                        self.agent.action_state_machine.roam()
                        break
                    # if not, search again by rotating
                    else:
                        self.agent.action_state_machine.search()
                        break
                # if exploratory_direction is not zero, keep rotating.
                else:
                    self.agent.action_state_machine.hold()
                    break

            # if the state is roaming
            elif self.agent.action_state_machine.is_roaming:
                # if find a target while roaming, then chase it
                if self.agent.perception.is_target_found():
                    self.agent.action_state_machine.chase()
                    break
                # if can go in the current direction, then go
                elif self.agent.perception.is_front_safe() and not self.agent.perception.is_nearly_static():
                    self.agent.action_state_machine.hold()
                    break
                # if not, stop
                else:
                    self.agent.action_state_machine.decelerate()
                    break

            # if the state is chasing
            elif self.agent.action_state_machine.is_chasing:
                # if a positive reward is received, then current chase is done
                if self.agent.perception.is_chasing_done():
                    # if there exists more things to chase, then chase one
                    if self.agent.perception.is_target_found():
                        self.agent.action_state_machine.chase()
                        break
                    # if not, stop
                    else:
                        self.agent.action_state_machine.decelerate()
                        break
                # chase can be failed. If so, stop
                elif self.agent.chase_failed:
                    self.agent.action_state_machine.decelerate()
                    break
                # if chase is not done, and not failed
                else:
                    # keep chasing
                    if self.agent.not_seeing_target_step_n < AgentConstants.not_seeing_target_step_limit:
                        self.agent.action_state_machine.hold()
                        break
                    # if the agent has lost the visual of the target for too long, then stop chasing
                    else:
                        self.agent.action_state_machine.decelerate()
                        break

            # if the state is decelerating (stopping)
            elif self.agent.action_state_machine.is_decelerating:
                # again, if found a target, then chase it.
                if self.agent.perception.is_target_found():
                    self.agent.action_state_machine.chase()
                    break
                # if it has come to a full stop, then start searching again
                elif self.agent.perception.is_static():
                    self.agent.action_state_machine.search()
                    break
                # if none above, keep decelerating.
                else:
                    self.agent.action_state_machine.hold()
                    break

            # if the state is in an unknown state (this won't be possible, simply for debug purposes)
            else:
                warnings.warn("An unknown state: {}".format(self.agent.action_state_machine.current_state))
                self.agent.action_state_machine.reset()
                break

        # Although the strategy seems a complete one,
        # the agent still can get stuck somewhere.
        # This line detects this scenario and helps it move out of it.
        self.deadlock_breaker()

    def deadlock_breaker(self):
        '''
        This method detects the situation where the agent gets stuck
        and helps it move out of it, by counting the steps in which the agent
        stays where it is (rotating won't change its position)
        :return:
        '''

        if self.agent.action_state_machine.is_chasing:

            if self.agent.perception.is_nearly_static():
                self.agent.static_step_n += 1
            else:
                self.agent.static_step_n = 0

            # if the agent hasn't moved for too long, force it to rotate to a random diection
            # and go in that direction
            if self.agent.static_step_n > AgentConstants.deadlock_step_limit:
                self.agent.exploratory_direction = np.random.choice(AgentConstants.directions_for_deadlock)
                self.agent.action_state_machine.rotate_to_direction()
                self.agent.static_step_n = 0
        else:
            self.agent.static_step_n = 0

    def reset(self):
        pass
