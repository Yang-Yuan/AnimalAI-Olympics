import numpy as np
from skimage import measure
import agentUtils
from ActionStateMachine import ActionStateMachine
from strategy import Strategy
from perception import Perception
import AgentConstants
import queue


class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """

        self.t = None
        self.step_n = None

        # functional modules
        self.actionStateMachine = ActionStateMachine(self)
        self.strategy = Strategy(self)
        self.perception = Perception(self)

        # primitive perception
        self.obs_visual = None
        self.obs_vector = None
        self.obs_visual_h = None
        self.done = None
        self.reward = None
        self.info = None

        # precess perceptions
        self.is_green = None
        self.is_brown = None
        self.is_red = None
        self.is_orange = None
        self.is_yellow = None
        self.is_target_color = None

        # memory
        self.visual_h_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.is_green_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.is_brown_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.is_red_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.is_orange_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.is_yellow_memory = queue.Queue(maxsize = AgentConstants.memory_size)
        self.vector_memory = queue.Queue(maxsize = AgentConstants.memory_size)

        # output action
        self.currentAction = None

        # strategy-related variables
        self.pirouette_step_n = None
        self.target_color = None
        self.safest_direction = None
        # TODO self.visual_imagery reconstruct mental imagery from primitive perception

    def reset(self, t):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """

        self.t = t
        self.step_n = 0

        # functional modules
        self.actionStateMachine.reset()
        self.strategy.reset()
        self.perception.reset()

        # primitive perception
        self.obs_visual = None
        self.obs_vector = None
        self.obs_visual_h = None
        self.done = None
        self.reward = None
        self.info = None

        # precess perceptions
        self.is_green = None
        self.is_brown = None
        self.is_red = None
        self.is_orange = None
        self.is_yellow = None
        self.is_target_color = None

        # memory
        self.visual_h_memory.queue.clear()
        self.is_green_memory.queue.clear()
        self.is_brown_memory.queue.clear()
        self.is_red_memory.queue.clear()
        self.is_orange_memory.queue.clear()
        self.is_yellow_memory.queue.clear()
        self.vector_memory.queue.clear()

        # output action
        self.currentAction = None

        # strategy-related variables
        self.pirouette_step_n = None
        self.target_color = None
        self.safest_direction = None

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current state of the environment
        We will run the Gym environment (AnimalAIEnv) and pass the arguments returned by env.step() to
        the agent.

        Note that should if you prefer using the BrainInfo object that is usually returned by the Unity
        environment, it can be accessed from info['brain_info'].

        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including eBrainInfo.
        :return: the action to take, a list or size 2
        """

        # set primitive observations
        self.obs_visual, self.obs_vector = obs
        self.obs_visual_h = agentUtils.toHue(self.obs_visual)
        self.done = done
        self.reward = reward
        self.info = info

        # perceive atop primtive observations
        self.perception.perceive()

        # run strategy
        self.strategy.run_strategy()

        return self.currentAction

        diff_green = abs(obs_visual_h - AgentConstants.predefined_colors_h.get("green"))
        is_green = diff_green < AgentConstants.color_diff_limit

        self.step_n += 1

        if not is_green.any():
            self.pirouette_step_n += 1
            return [0, 1]

        if 1 == is_green.sum():
            diff_center = np.array(np.where(is_green)).transpose()[0] - AgentConstants.center_of_view
            target_size = 1
        else:
            labels, label_num = measure.label(input=is_green, background=False, return_num=True, connectivity=1)
            sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
            target_label = np.argmax(sizes) + 1
            center_of_target = np.array(np.where(labels == target_label)).mean(axis=1)
            diff_center = center_of_target - AgentConstants.center_of_view
            target_size = sizes[target_label - 1]

        if diff_center[1] < -AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                self.pirouette_step_n = 0
                return [1, 2]
            else:
                self.pirouette_step_n += 1
                return [0, 2]
        elif diff_center[1] > AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                self.pirouette_step_n = 0
                return [1, 1]
            else:
                self.pirouette_step_n += 1
                return [0, 1]
        else:
            self.pirouette_step_n = 0
            return [1, 0]
