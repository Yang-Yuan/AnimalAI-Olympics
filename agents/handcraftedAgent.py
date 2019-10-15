import numpy as np
from skimage import measure
import agentUtils


class Agent(object):

    # Task-related constants
    predefined_colors = {"green": [0.506, 0.749, 0.255],
                         "brown": [0.471, 0.337, 0.0471],
                         "red": [0.722, 0.196, 0.196],
                         "orange": [1, 0.675, 0.282],
                         "box_dark": [0.196, 0.165, 0.133],
                         "box_light": [0.318, 0.267, 0.22],
                         "UL": [0.435, 0.367, 0.2]}
    predefined_colors_h = {k: agentUtils.toHue(v.reshape(1, 1, 3))[0, 0] \
                           for (k, v) in predefined_colors.items()}

    # Environmental constants
    resolution = 84
    center_of_view = [resolution / 2, resolution / 2]
    default_test_length = 1000

    # Perception limits of colors
    color_diff_limit = 0.075 # TODO different limits for different object

    # Control constants
    aim_error_limit = 5
    size_limit = 5
    hl = 2



    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """
        self.t = 0
        self.step_n = 0
        self.total_reward = 0
        self.pirouette_step_n = 0
        self.diff_center_old = None
        self.target_size_old = None
        self.visual_memory = None

        self.visual_imagery = np.zeros((Agent.resolution, Agent.resolution))

        self.cluster_pixel_idx = None
        self.cluster_bin_idx = None
        self.cluster_centers = None

        self.bin_sizes = None

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.t = t
        self.step_n = 0
        self.total_reward = 0
        self.pirouette_step_n = 0
        self.visual_memory = np.zeros((self.t, Agent.resolution, Agent.resolution), dtype=np.int)

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
        if done:
            return [0, 0]

        # magic code!
        if self.pirouette_step_n > 70:
            self.pirouette_step_n = 0
            return [1, 0]

        obs_visual, obs_vector = obs
        obs_visual_h = Agent.toHue(obs_visual)

        self.histogramize(obs_visual_h)

        diff_green = abs(obs_visual_h - Agent.predefined_colors_h.get("green"))
        is_green = diff_green < Agent.color_diff_limit

        self.step_n += 1

        if not is_green.any():
            self.pirouette_step_n += 1
            return [0, 1]

        if 1 == is_green.sum():
            diff_center = np.array(np.where(is_green)).transpose()[0] - Agent.center_of_view
            target_size = 1
        else:
            labels, label_num = measure.label(input=is_green, background=False, return_num=True, connectivity=1)
            sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
            target_label = np.argmax(sizes) + 1
            center_of_target = np.array(np.where(labels == target_label)).mean(axis=1)
            diff_center = center_of_target - Agent.center_of_view
            target_size = sizes[target_label - 1]

        if diff_center[1] < -Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            if target_size < Agent.size_limit:
                self.pirouette_step_n = 0
                return [1, 2]
            else:
                self.pirouette_step_n += 1
                return [0, 2]
        elif diff_center[1] > Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            if target_size < Agent.size_limit:
                self.pirouette_step_n = 0
                return [1, 1]
            else:
                self.pirouette_step_n += 1
                return [0, 1]
        else:
            self.pirouette_step_n = 0
            return [1, 0]
