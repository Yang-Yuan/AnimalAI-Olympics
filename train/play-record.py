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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("play-record-logger")

##################################################
# Specify the arena config here
##################################################
arenaConfig = '../configs/234-POA/poa-3.yaml'

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


##################################################
# Add a keyboard listener
##################################################
def on_press(key):
    global queueLR
    global queueFB

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
# Simulation Utilities
##################################################
def getAction():
    lr = None
    fb = None

    global queueLR
    global queueFB

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


def record(visual, action):
    global visuals
    global actions
    global step

    if visuals.shape[0] == step:
        visuals = np.concatenate((visuals,
                                  np.zeros((INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels),
                                           dtype=np.uint8)),
                                 axis=0)
        actions = np.concatenate((actions, np.zeros((INCREMENT_MEMORY_SIZE, dim_actions * n_arenas), dtype=np.uint8)),
                                 axis=0)

    visuals[step + 1, :, :, :, :] = (visual * 255).astype(dtype=np.uint8)
    actions[step, :] = action.astype(dtype=np.uint8)


def save():
    global visuals
    global actions
    global step
    global directoryName

    visuals = visuals[range(step + 1), :, :, :, :]
    actions = actions[range(step + 1), :]

    fileName = directoryName + "/" + str(uuid.uuid4())
    np.savez(fileName, visuals = visuals, actions = actions)


def restart():
    global visuals
    global actions
    global step
    global queueLR
    global queueFB
    global brainInfo

    step = 0
    visuals = np.zeros(shape=(INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels), dtype=np.uint8)
    actions = np.zeros(shape=(INITIAL_MEMORY_SIZE, dim_actions * n_arenas), dtype=np.uint8)
    visuals[step, :, :, :, :] = (brainInfo['Learner'].visual_observations[0] * 255).astype(dtype=np.uint8)

    while not queueLR.empty():
        try:
            queueLR.get_nowait()
        except queue.Empty:
            logger.info("reseted queueLR")

    while not queueFB.empty():
        try:
            queueFB.get_nowait()
        except queue.Empty:
            logger.info("reset queueFB")


def run_step():
    global step
    global brainInfo

    action = getAction()

    brainInfo = env.step(action)

    record(brainInfo['Learner'].visual_observations[0], action)

    if all(brainInfo['Learner'].local_done):
        save()
        brainInfo = env.reset()
        restart()
    else:
        step += 1


##################################################
# Global variables to maintain in simulation
##################################################
step = 0
queueFB = queue.Queue(maxsize=5)
queueLR = queue.Queue(maxsize=5)
visuals = np.zeros(shape=(INITIAL_MEMORY_SIZE, n_arenas, resolution, resolution, n_channels), dtype=np.uint8)
actions = np.zeros(shape=(INITIAL_MEMORY_SIZE, dim_actions * n_arenas), dtype=np.uint8)
brainInfo = env.reset(arenas_configurations=arena_config_in)
visuals[step, :, :, :, :] = (brainInfo['Learner'].visual_observations[0] * 255).astype(dtype=np.uint8)

plt.ion()
fig, ax = plt.subplots()
image = ax.imshow(np.zeros((resolution, resolution, 3)))


##################################################
# Play and record
##################################################
try:
    while True:
        image.set_data(brainInfo['Learner'].visual_observations[0][0, :, :, :])
        fig.suptitle('Step = ' + str(step))
        fig.canvas.draw()
        fig.canvas.flush_events()
        run_step()
except Exception as e:
    logger.debug(e)
finally:
    env.close()
    listener.stop()
