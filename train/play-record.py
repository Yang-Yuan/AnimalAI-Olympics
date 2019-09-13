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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("play-record-logger")

##################################################
# Specify the arena config here
##################################################
arenaConfig = '../examples/configs/1-Food.yaml'

##################################################
# Create a folder to store data of this task
##################################################
match = re.search('/([0-9A-Za-z\-]+?)\.yaml', arenaConfig)
directoryName = None
if match:
    directoryName = './' + match.group(1)
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
else:
    raise NameError("Can't find task name, man.")

##################################################
# So many parameter, seriously?
##################################################
INITIAL_MEMORY_SIZE = 2000
INCREMENT_MEMORY_SIZE = 250

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
resolution = 84
n_channels = 3
dim_actions = 2

##################################################
# Start up unity environment.
##################################################
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

arena_config_in = ArenaConfig(arenaConfig)
info = env.reset(arenas_configurations=arena_config_in)

##################################################
# Initialize the numpy arrays for recording
##################################################
visuals = np.zeros(shape = (INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels), dtype=np.uint8)
actions = np.zeros(shape = (INITIAL_MEMORY_SIZE, dim_actions * n_arenas), dtype = np.uint8)
visuals[0, :, :, :, :] = info['Learner'].visual_observations[0]

##################################################
# Add a keyboard listener
##################################################
queueFB = queue.Queue(maxsize=5)
queueLR = queue.Queue(maxsize=5)


def on_press(key):
    try:
        logger.debug('alphanumeric key {0} pressed'.format(
            key.char))
        if utils.equalIgnoreCase(key.char, 'W'):
            utils.insertQ(queueFB, 1)
        elif utils.equalIgnoreCase(key.char, 'S'):
            utils.insertQ(queueFB, 2)
        elif utils.equalIgnoreCase(key.char, 'A'):
            utils.insertQ(queueLR, 2)
        elif utils.equalIgnoreCase(key.char, 'D'):
            utils.insertQ(queueLR, 1)
        elif utils.equalIgnoreCase(key.char, 'N'):
            utils.insertQ(queueLR, 0)
            utils.insertQ(queueFB, 0)
    except Exception as e:
        logger.debug(e)


listener = keyboard.Listener(on_press=on_press)
listener.start()

##################################################
# Visualization
##################################################
plt.ion()
fig, ax = plt.subplots()
image = ax.imshow(info['Learner'].visual_observations[0][0, :, :, :])
fig.canvas.draw()
fig.canvas.flush_events()


def getAction():
    lr = None
    fb = None

    # retry
    while True:

        try:
            lr = queueLR.get_nowait()
        except Exception as e:
            logger.debug(e)

        try:
            fb = queueFB.get_nowait()
        except Exception as e:
            logger.debug(e)

        if (lr is not None) or (fb is not None):
            break

    lr = 0 if lr is None else lr
    fb = 0 if fb is None else fb
    return np.array([fb, lr])


def record(step, visual, action):
    global visuals
    global actions

    if visuals.shape[0] == step:
        visuals = np.concatenate((visuals,
                                  np.zeros((INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels),
                                           dtype=np.uint8)),
                                 axis=0)
        actions = np.concatenate((actions, np.zeros((INCREMENT_MEMORY_SIZE, dim_actions * n_arenas), dtype=np.uint8)),
                                 axis=0)

    visuals[step + 1, :, :, :, :] = visual.astype(dtype = np.uint8)
    actions[step, :] = action.astype(dtype = np.uint8)


def saveAndRestart(info):
    global visuals
    global actions
    global step

    visuals = visuals[range(step + 2), :, :, :, :]
    actions = visuals[range(step + 1), :]

    fileName = directoryName + "/" + uuid.uuid4()
    np.savez(fileName, visuals, actions)

    visuals = np.zeros(shape=(INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels), dtype=np.uint8)
    actions = np.zeros(shape=(INITIAL_MEMORY_SIZE, dim_actions * n_arenas), dtype=np.uint8)
    visuals[0, :, :, :, :] = info['Learner'].visual_observations[0]
    step = 0

def run_step(step):
    action = getAction()

    res = env.step(action)
    fig.suptitle('Step = ' + str(step))
    image.set_data(res['Learner'].visual_observations[0][0, :, :, :])

    record(step, res['Learner'].visual_observations[0], action)

    if all(res['Learner'].local_done):
        brainInfo = env.reset()
        saveAndRestart(brainInfo)

step = 0
try:
    while True:
        run_step(step)
        step += 1
        fig.canvas.draw()
        fig.canvas.flush_events()
except Exception as e:
    logger.debug(e)
finally:
    env.close()
    listener.stop()
