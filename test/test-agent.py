from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import logging
import random
import numpy as np
from matplotlib import pyplot as plt
import re
import queue
from pynput import keyboard
import utils
import os
import uuid
import constants
import time
from skimage import segmentation, color
from scipy.stats import binned_statistic

from handcraftedAgent import Agent

# arenaConfigs = ['../examples/configs/1-Food.yaml',
#                 '../examples/configs/2-Preferences.yaml',
#                 '../examples/configs/3-Obstacles.yaml',
#                 '../examples/configs/4-Avoidance.yaml',
#                 '../examples/configs/5-SpatialReasoning.yaml',
#                 '../examples/configs/6-Generalization.yaml',
#                 '../examples/configs/7-InternalMemory.yaml']

arenaConfigs = ['../configs/1-Food/single-static.yaml',
                '../configs/1-Food/two-static.yaml',
                '../configs/1-Food/three-static.yaml',
                '../configs/1-Food/multi-static.yaml',
                '../configs/1-Food/single-dynamic.yaml',
                '../configs/1-Food/two-dynamic.yaml',
                '../configs/1-Food/three-dynamic.yaml',
                '../configs/1-Food/multi-dynamic.yaml',
                '../configs/1-Food/single-mix.yaml',
                '../configs/1-Food/two-mix.yaml',
                '../configs/1-Food/three-mix.yaml',
                '../configs/1-Food/multi-mix.yaml']

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)

seed = 333
base_port = 5005
sub_id = 1
run_id = 'train_example'
run_seed = 1
docker_target_name = None
no_graphics = False
n_arenas = 1
resolution = constants.resolution
n_channels = constants.n_channels
dim_actions = constants.dim_actions

if env_path is not None:
    env_path = (env_path.strip()
                .replace('.app', '')
                .replace('.exe', '')
                .replace('.x86_64', '')
                .replace('.x86', ''))
docker_training = docker_target_name is not None

env = UnityEnvironment(
    n_arenas=n_arenas,
    file_name=env_path,
    worker_id=worker_id,
    seed=seed,
    docker_training=docker_training,
    play=False,
    resolution=resolution
)

n_bins = 30
bin_edges = np.linspace(start = 0, stop = 1, num = n_bins + 1)

plt.ion()
fig, axs = plt.subplots(ncols = 2, nrows = 1)
image = axs[0].imshow(np.zeros((resolution, resolution, 3)))
bars = axs[1].bar(x = (bin_edges[ : -1] + bin_edges[1 : ]) / 2,
                  height = np.repeat(n_bins, n_bins),
                  width = 1 / n_bins,
                  bottom = 0)

agent = Agent()

for arenaConfig in arenaConfigs:
    print(arenaConfig)
    arena_config_in = ArenaConfig(arenaConfig)
    for sample_n in range(constants.sample_size_per_task):
        print("ArenaConfig; {} Sample: {}".format(arenaConfig, sample_n))
        agent.reset(arena_config_in.arenas[0].t)
        brainInfo = env.reset(arenas_configurations=arena_config_in)

        while True:
            obs = brainInfo['Learner'].visual_observations[0][0, :, :, :], brainInfo['Learner'].vector_observations
            reward = brainInfo['Learner'].rewards[0]
            done = brainInfo['Learner'].local_done[0]
            info = {"brain_info": brainInfo}

            # if sample_n == 56:
            image.set_data(obs[0])
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # seg = segmentation.felzenszwalb(color.rgb2hsv(obs[0]), scale = 500)
            # seg = segmentation.quickshift(color.rgb2hsv(obs[0]), ratio = 1, kernel_size = 100)
            # image.set_data(segmentation.mark_boundaries(obs[0], seg))
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            obs_hsv = color.rgb2hsv(obs[0])
            H, bin_edges = np.histogram(a = obs_hsv[:, :, 0], bins = n_bins, range = (0, 1), density = True)
            bin_labels = np.digitize(obs_hsv[:, :, 0], bin_edges)
            bin_colors = np.zeros((n_bins, 3))
            for bin_label in range(1, n_bins):
                bin_pixel = obs[0][np.where(bin_labels == bin_label)]
                if 0 != len(bin_pixel):
                    bin_colors[bin_label - 1] = bin_pixel.mean(axis = 0)

            for bar, height, face_color in zip(bars, H, bin_colors):
                bar.set_height(height)
                bar.set_facecolor(face_color)
            fig.canvas.draw()
            fig.canvas.flush_events()

            action = agent.step(obs, reward, done, info)

            if all(brainInfo['Learner'].local_done):
                break
            else:
                brainInfo = env.step(action)

env.close()

