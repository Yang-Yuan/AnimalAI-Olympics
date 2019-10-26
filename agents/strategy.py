import AgentConstants
import warnings
import sys

class Strategy(object):

    def __init__(self, agent):
        self.agent = agent

    def run_strategy(self):

        # if done
        if self.agent.done:
            self.agent.currentAction = AgentConstants.taxi
            return

        while True:

            # if the agent is static
            if self.agent.actionStateMachine.is_static:
                if 0 == self.agent.pirouette_step_n:
                    self.agent.actionStateMachine.pirouette()
                    break
                else:
                    warnings.warn("undefined branch!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    sys.exit(1)

            # if the agent is pirouetting
            elif self.agent.actionStateMachine.is_pirouetting:
                if self.agent.pirouette_step_n < AgentConstants.pirouette_step_limit:
                    self.agent.actionStateMachine.hold()
                    break
                else:
                    if self.agent.target_color is None:
                        if self.agent.safest_direction == 0:
                            if self.agent.perception.is_front_safe():
                                self.agent.actionStateMachine.roam()
                                break
                            else:
                                self.agent.actionStateMachine.pirouette()
                                break
                        else:
                            self.agent.actionStateMachine.rotate_to_direction()
                            break
                    else:
                        if self.agent.perception.is_found():
                            self.agent.actionStateMachine.chase()
                            break
                        else:
                            self.agent.actionStateMachine.search()
                            break

            # if the agent is rotating_to_direction
            elif self.agent.actionStateMachine.is_rotating_to_direction:
                if self.agent.safest_direction == 0:
                    if self.agent.perception.is_front_safe():
                        self.agent.actionStateMachine.roam()
                        break
                    else:
                        self.agent.actionStateMachine.pirouette()
                        break
                else:
                    if self.agent.perception.renew_target():
                        self.agent.actionStateMachine.chase()
                        break
                    else:
                        self.agent.actionStateMachine.hold()
                        break

            # if the agent is roaming
            elif self.agent.actionStateMachine.is_roaming:
                if self.agent.perception.renew_target():
                    self.agent.actionStateMachine.chase()
                    break
                elif self.agent.perception.is_front_safe() and not self.agent.perception.is_nearly_static():
                    self.agent.actionStateMachine.hold()
                    break
                else:
                    self.agent.actionStateMachine.decelerate()
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
                        if self.agent.target_color is None:
                            if self.agent.safest_direction == 0:
                                if self.agent.perception.is_front_safe():
                                    self.agent.actionStateMachine.roam()
                                    break
                                else:
                                    self.agent.actionStateMachine.pirouette()
                                    break
                            else:
                                self.agent.actionStateMachine.rotate_to_direction()
                                break
                        else:
                            if self.agent.perception.is_found():
                                self.agent.actionStateMachine.chase()
                                break
                            else:
                                self.agent.actionStateMachine.search()
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
                else:
                    if self.agent.not_seeing_target_step_n < AgentConstants.not_seeing_target_step_limit:
                        self.agent.actionStateMachine.hold()
                        break
                    else:
                        self.agent.actionStateMachine.decelerate()
                        break

            # if the agent is decelerating
            elif self.agent.actionStateMachine.is_decelerating:
                if self.agent.perception.is_static():
                    self.agent.actionStateMachine.pirouette()
                    break
                else:
                    self.agent.actionStateMachine.hold()
                    break

            # if the agent is in an unknown state
            else:
                warnings.warn("An unknown state: {}".format(self.agent.actionStateMachine.current_state))
                sys.exit(1)

    def reset(self):
        pass
