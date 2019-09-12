from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import logging
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import re
import queue
from pynput import keyboard
import utils
import os
import time

logging.basicConfig(level = logging.INFO)
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
MAXIMUM_STEPS = 2000

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
# Add a keyboard listener
##################################################
queueFB = queue.Queue()
queueLR = queue.Queue()


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
    except AttributeError:
        logger.debug('special key {0} pressed'.format(
            key))


listener = keyboard.Listener(on_press=on_press)
listener.start()

##################################################
# Visualization
##################################################
fig, ax = plt.subplots()
image = ax.imshow(np.zeros((resolution, resolution, 3)))


def initialize_animation():
    image.set_data(np.zeros((resolution, resolution, 3)))


def getAction():
    lr = 0
    fb = 0

    # retry for 5 times
    for _ in range(5):

        try:
            lr = queueLR.get_nowait()
        except Exception as e:
            logger.debug(e)

        try:
            fb = queueFB.get_nowait()
        except Exception as e:
            logger.debug(e)

        if lr != 0 or fb != 0:
            break
        else:
            time.sleep(0.005)

    return np.array([fb, lr])


def run_step_imshow(step):

    action = getAction()

    res = env.step(action)
    fig.suptitle('Step = ' + str(step))
    image.set_data(res['Learner'].visual_observations[0][0, :, :, :])

    if all(res['Learner'].local_done):
        env.reset()

    return image


try:
    anim = animation.FuncAnimation(fig, run_step_imshow, init_func=initialize_animation, interval=50, repeat=False)
    plt.show()
finally:
    env.close()
    keyboard.listener.stop()
