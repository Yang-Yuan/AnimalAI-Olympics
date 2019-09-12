from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import re
import os
import queue
import threading
import readchar

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
# Start a new thread for IO
##################################################
queueFB = queue.Queue()
queueLR = queue.Queue()


def equalIgnoreCase(a, b):
    return a.lower() == b.lower()


def insertQ(q, x):
    try:
        q.put_nowait(x)
    except Exception as e:
        print(e)


def inputReceiver():
    while True:
        char = readchar.readchar()
        if equalIgnoreCase('A', char):
            insertQ(queueLR, 1)
        elif equalIgnoreCase('D', char):
            insertQ(queueLR, 2)
        elif equalIgnoreCase('W', char):
            insertQ(queueFB, 1)
        elif equalIgnoreCase('S', char):
            insertQ(queueFB, 2)


receiver = threading.Thread(target = inputReceiver)
receiver.setDaemon(True)
receiver.start()


##################################################
# Visualization
##################################################
fig, ax = plt.subplots()
ax.axis("off")
image = ax.imshow(np.zeros((resolution, resolution, 3)))


def initialize_animation():
    image.set_data(np.zeros((resolution, resolution, 3)))

def getAction():
    lr = 0
    fb = 0
    try:
        lr = queueLR.get_nowait()
    except Exception as e:
        print(e)

    try:
        fb = queueFB.get_nowait()
    except Exception as e:
        print(e)

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
