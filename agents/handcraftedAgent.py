import numpy as np
import random
import sys
from scipy.spatial.distance import pdist, cosine
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


class Agent(object):

    green = [0.506, 0.749, 0.255]
    color_diff_limit = 0.01
    position_diff_limit = 1
    center_of_view = [41.5, 41.5]
    aim_error_limit = 5
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
        if done:
            return [0, 0]

        if 250 == self.step_n:
            print("Failed")
            sys.exit(1)

        if self.pirouette_step_n > 60:
            self.pirouette_step_n = 0
            return [random.randint(1, 2), random.randint(1, 2)]

        self.total_reward += reward
        # print("step:{} reward:{} total_reward:{} done:{}".format(self.step_n, reward, self.total_reward, done))
        self.step_n += 1

        # is_green = abs((obs - Agent.green)).sum(axis=2) < Agent.color_diff_limit
        is_green = np.zeros(obs.shape[0 : 2], dtype = bool)
        for ii in range(obs.shape[0]):
            for jj in range(obs.shape[1]):
                is_green[ii, jj] = cosine(obs[ii, jj], Agent.green) < Agent.color_diff_limit

        if is_green.any():
            self.pirouette_step_n = 0
            ind_green = np.where(is_green)
        else:
            self.pirouette_step_n += 1
            return [0, 1]

        X = np.array(ind_green).transpose()

        if 1 == len(X):
            diff_center = X[0] - Agent.center_of_view
            target_size = 1
        else:
            dist_x = pdist(X, 'cityblock')
            link_x = linkage(y=dist_x, method="single", optimal_ordering=True)
            cluster_label_x = fcluster(link_x, Agent.position_diff_limit, 'distance')
            cluster_labels = np.unique(cluster_label_x)
            cluster_sizes = [(cluster_label_x == cluster_label).sum() for cluster_label in cluster_labels]
            largest_cluster_label = cluster_labels[np.argmax(cluster_sizes)]
            largest_cluster = X[np.where(cluster_label_x == largest_cluster_label)]
            center_of_the_largest = largest_cluster.mean(axis=0)
            diff_center = center_of_the_largest - Agent.center_of_view
            target_size = len(largest_cluster)

        if diff_center[1] < -Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            return [0, 2]
        elif diff_center[1] > Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            return [0, 1]
        else:
            return [1, 0]
