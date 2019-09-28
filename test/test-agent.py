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

from handcraftedAgent import Agent

arenaConfigs = ['../examples/configs/1-Food.yaml',
                '../examples/configs/2-Preferences.yaml',
                '../examples/configs/3-Obstacles.yaml',
                '../examples/configs/4-Avoidance.yaml',
                '../examples/configs/5-SpatialReasoning.yaml',
                '../examples/configs/6-Generalization.yaml',
                '../examples/configs/7-InternalMemory.yaml']

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)

seed = 10
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

for arenaConfig in arenaConfigs:
    arena_config_in = ArenaConfig(arenaConfig)
    for _ in range(constants.sample_size_per_task):
        agent.reset(arena_config_in.arenas[0].t)
        brainInfo = env.reset(arenas_configurations=arena_config_in)

        while True:
            obs = brainInfo['Learner'].visual_observations[0][0, :, :, :]
            reward = brainInfo['Learner'].rewards[0]
            done = brainInfo['Learner'].local_done[0]
            info = {"brain_info": brainInfo}
            action = agent.step(obs, reward, done, info)

            if all(brainInfo['Learner'].local_done):
                break
            else:
                brainInfo = env.step(action)

env.close()
