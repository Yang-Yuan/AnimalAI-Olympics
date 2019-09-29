import numpy as np


class Agent(object):

    green = [0.506, 0.749, 0.255]
    color_diff_limit = 0.1

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """
        pass

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.t = t
        self.step_n = 0
        self.total_reward = 0


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
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """
        self.total_reward += reward
        # print("step:{} reward:{} total_reward:{} done:{}".format(self.step_n, reward, self.total_reward, done))
        self.step_n += 1

        diff_green = abs((obs - Agent.green).sum(axis=2))
        ind_min = np.unravel_index(diff_green.argmin(axis=None), diff_green.shape)
        diff_min = diff_green[ind_min]

        green_clusters = None
        if diff_min > Agent.color_diff_limit:
            return [0, 1]
        else:
            green_clusters = [[ind_min]]

