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
import AgentConstants

from handcraftedAgent import Agent

# arenaConfigs = [#'../examples/configs/1-Food.yaml',
#                 '../examples/configs/2-Preferences.yaml']
                # '../examples/configs/3-Obstacles.yaml',
                # '../examples/configs/4-Avoidance.yaml',
                # '../examples/configs/5-SpatialReasoning.yaml',
                # '../examples/configs/6-Generalization.yaml',
                # '../examples/configs/7-InternalMemory.yaml']

# arenaConfigs = ['../configs/1-Food/single-static.yaml',
#                 '../configs/1-Food/two-static.yaml',
#                 '../configs/1-Food/three-static.yaml',
#                 '../configs/1-Food/multi-static.yaml',
#                 '../configs/1-Food/single-dynamic.yaml',
#                 '../configs/1-Food/two-dynamic.yaml',
#                 '../configs/1-Food/three-dynamic.yaml',
#                 '../configs/1-Food/multi-dynamic.yaml',
#                 '../configs/1-Food/single-mix.yaml',
#                 '../configs/1-Food/two-mix.yaml',
#                 '../configs/1-Food/three-mix.yaml',
#                 '../configs/1-Food/multi-mix.yaml']

arenaConfigs = [#'../configs/234-POA/poa-1.yaml',
                #'../configs/234-POA/poa-2.yaml',
                '../configs/234-POA/poa-3.yaml']

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)

seed = 333
np.random.seed(333)
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
fig, ax = plt.subplots(ncols=1, nrows=1)
image = ax.imshow(np.zeros((resolution, resolution, 3)))
line, = ax.plot([], [])
sca = ax.scatter([], [], s = 5, c="yellow")


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

            action = agent.step(obs, reward, done, info)

            # Visualization
            image.set_data(obs[0])
            if agent.chaser.newest_path is not None:
                sca.set_offsets(np.array(agent.chaser.newest_path))
            else:
                sca.set_offsets(AgentConstants.standpoint[::-1])
            if agent.chaser.newest_end is not None:
                line.set_xdata([AgentConstants.standpoint[1], agent.chaser.newest_end[0]])
                line.set_ydata([AgentConstants.standpoint[0], agent.chaser.newest_end[1]])
            else:
                line.set_xdata([])
                line.set_ydata([])
            fig.canvas.draw()
            fig.canvas.flush_events()

            if all(brainInfo['Learner'].local_done):
                break
            else:
                brainInfo = env.step(action)

plt.close(fig)
env.close()
