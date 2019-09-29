import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


class Agent(object):

    green = [0.506, 0.749, 0.255]
    color_diff_limit = 0.1
    position_diff_limit = 1

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

        is_green = abs((obs - Agent.green)).sum(axis=2) < Agent.color_diff_limit
        if is_green.any():
            ind_green = np.where(is_green)
        else:
            return [0, 1]

        X = np.array(ind_green).transpose()
        # clusters_x = np.arange(len(X)).reshape(-1, 1)
        # dist_x = np.zeros((len(x), len(x)))
        # for ii in range(len(x)):
        #     for jj in range(ii + 1, len(x)):
        #         dist_x[ii, jj] = dist_x[jj, ii] = sum(abs(x[ii] - x[jj]))
        # dist_x[np.where(dist_x > Agent.position_diff_limit)] = float('inf')

        dist_x = pdist(X, 'cityblock')
        link_x = linkage(y=dist_x, method="single", optimal_ordering=True)
        cluster_label_x = fcluster(link_x, Agent.position_diff_limit, 'distance')
        cluster_labels = np.unique(cluster_label_x)
        cluster_sizes = [(cluster_label_x == cluster_label).sum() for cluster_label in cluster_labels]
        largest_cluster_label = cluster_labels[np.argmax(cluster_sizes)]
        largest_cluster = X[np.where(cluster_label_x == largest_cluster_label)]
        center_of_the_largest = largest_cluster.mean(axis = 0)



        # diff_green = abs((obs - Agent.green).sum(axis=2))
        # ind_min = np.unravel_index(diff_green.argmin(axis=None), diff_green.shape)
        # diff_min = diff_green[ind_min]
        #
        # green_clusters = None
        # if diff_min > Agent.color_diff_limit:
        #     return [0, 1]
        # else:
        #     green_points = np.array(ind_min).reshape(1, 2)
        #     while True:
        #         diff_green[ind_min] = float("inf")
        #         ind_min = np.unravel_index(diff_green.argmin(axis=None), diff_green.shape)
        #         diff_min = diff_green[ind_min]
        #         dist_min = abs(green_points - np.array(ind_min)).sum(axis = 1).min()
        #         if diff_min < Agent.color_diff_limit and
        return [0, 0]