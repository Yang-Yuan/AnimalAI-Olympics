from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import random
import numpy as np
from matplotlib import pyplot as plt

# import my handcrafted agent
from handcraftedAgent import Agent
import AgentConstants

# arena config files used to test my handcrafted agent
arenaConfigs = [#'../configs/234-POA/poa-1.yaml',
                #'../configs/234-POA/poa-2.yaml',
                '../configs/234-POA/poa-3.yaml',
                '../configs/234-POA/poa-4.yaml',
                '../configs/1-Food/single-static.yaml',
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
                '../configs/1-Food/multi-mix.yaml',
                '../examples/configs/1-Food.yaml',
                '../examples/configs/2-Preferences.yaml',
                '../examples/configs/3-Obstacles.yaml',
                '../examples/configs/4-Avoidance.yaml',
                '../examples/configs/5-SpatialReasoning.yaml',
                '../examples/configs/6-Generalization.yaml',
                '../examples/configs/7-InternalMemory.yaml']

# parameters for setting up the testing env
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
resolution = 84
n_channels = 3
dim_actions = 2
sample_size_per_task = 30

# set up the testing env
np.random.seed(seed)
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

# The Agent to test
agent = Agent()

# visualization
plt.ion()
fig, ax = plt.subplots(ncols=1, nrows=1)
image = ax.imshow(np.zeros((resolution, resolution, 3))) # visual input for the agent
line, = ax.plot([], []) # the direction that the agent want to go given the visual input
sca = ax.scatter([], [], s = 5, c="yellow") # the path to the target(food)


# loop over all the config files above
for arenaConfig in arenaConfigs:
    print(arenaConfig)
    arena_config_in = ArenaConfig(arenaConfig)

    # run multiple tests derived from each config file
    for sample_n in range(sample_size_per_task):
        print("ArenaConfig; {} Sample: {}".format(arenaConfig, sample_n))

        # initialize(reset) the agent and the env
        agent.reset(arena_config_in.arenas[0].t)
        brainInfo = env.reset(arenas_configurations=arena_config_in)

        while True:

            # information given by the env
            obs = brainInfo['Learner'].visual_observations[0][0, :, :, :], brainInfo['Learner'].vector_observations[0]
            reward = brainInfo['Learner'].rewards[0]
            done = brainInfo['Learner'].local_done[0]
            info = {"brain_info": brainInfo}

            # let the agent generate an action based on the information
            action = agent.step(obs, reward, done, info)

            # Visualization{visual, direction, path}
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

            # go to next test if the current one is finised
            if all(brainInfo['Learner'].local_done):
                break
            else:
                brainInfo = env.step(action)

# cleanup
plt.close(fig)
env.close()
