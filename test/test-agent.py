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

agent = Agent()

plt.ion()
fig, axs = plt.subplots(ncols = 2, nrows = 1)
image = axs[0].imshow(np.zeros((resolution, resolution, 3)))
bars = axs[1].bar(x = (agent.bin_edges[ : -1] + agent.bin_edges[1 : ]) / 2,
                  height = np.repeat(agent.resolution * agent.resolution, agent.n_bins),
                  width = agent.bin_length,
                  bottom = 0)
# visual_imagery = axs[2].imshow(np.zeros((resolution, resolution, 3)))

# fig_tmp, ax_tmp = plt.subplots()
# image_tmp = ax_tmp.imshow(np.zeros((resolution, resolution, 3)))


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

            # ax_tmp.imshow(obs[0])
            # fig_tmp.savefig("tmp.png")

            # image.set_data(obs[0])
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            action = agent.step(obs, reward, done, info)

            # seg = segmentation.felzenszwalb(color.rgb2hsv(obs[0]), scale = 500)
            # seg = segmentation.slic(color.rgb2hsv(obs[0]))
            # seg = segmentation.quickshift(color.rgb2hsv(obs[0]), ratio = 1, kernel_size = 100)
            # image_tmp.set_data(segmentation.mark_boundaries(obs[0], seg))
            # fig_tmp.canvas.draw()
            # fig_tmp.canvas.flush_events()

            # Visualization
            image.set_data(obs[0])
            for bar, height, face_color, idx in zip(bars, agent.bin_sizes, agent.bin_colors, agent.bin_pixel_idx):
                bar.set_height(height)
                bar.set_facecolor(plt.cm.hsv(face_color))
                # bar.set_facecolor(obs[0][tuple(idx)].mean(axis = 0))
            # visual_imagery.set_data(plt.cm.hsv(agent.visual_imagery))
            fig.canvas.draw()
            fig.canvas.flush_events()

            if all(brainInfo['Learner'].local_done):
                break
            else:
                brainInfo = env.step(action)

plt.close(fig)
# plt.close(fig_tmp)
env.close()

