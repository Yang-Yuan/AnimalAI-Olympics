from ActionStateMachine import ActionStateMachine
from strategy import Strategy
from perception import Perception
from chaser import Chaser
import AgentConstants
import queue
from skimage.color import rgb2hsv


class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """

        # functional modules
        self.perception = Perception(self)  # generate high-level perception based on current observation
        self.strategy = Strategy(self)  # implement strategy to chase the food given the high-level perception
        self.actionStateMachine = ActionStateMachine(self)  # use a finite state machine to help implement the strategy
        self.chaser = Chaser(self)  # responsible for chase the food

        # primitive perception
        self.obs_visual = None  # visual input in RGB space, float numpy array of shape (84, 84, 3)
        self.obs_visual_hsv = None  # visual input in HSV space, float numpy array of shape (84, 84, 3)
        self.obs_vector = None  # speed vector (left-right, up-down, forward-backward), float numpy array of shape (3,)
        self.done = None  # if the current test is done, bool
        self.reward = None  # current reward from the env, float
        self.info = None  # the brainInfo object from the env
        self.t = None  # how long will this test last? int
        self.step_n = None  # the current time step, int

        # high-level perceptions
        self.is_green = None  # bool numpy array of shape (84, 84), to indicate if each pixel is green (food color)
        self.is_brown = None  # bool numpy array of shape (84, 84), to indicate if each pixel is brown (food color)
        self.is_red = None  # bool numpy array of shape (84, 84), to indicate if each pixel is red (danger color)
        self.is_yellow = None  # bool numpy array of shape (84, 84), to indicate if each pixel is yellow (ground color)
        self.is_blue = None  # bool numpy array of shape (84, 84), to indicate if each pixel is blue (sky color)
        self.is_gray = None  # bool numpy array of shape (84, 84), to indicate if each pixel is gray (wall color)
        self.target_color = None  # the color is currently looking for, either brown or green
        self.is_inaccessible = None  # bool numpy array of shape (84, 84), to indicate if each pixel is inaccessible
                                     # because it might be a wall pixel, sky pixel or a dangerous pixel.
        self.is_inaccessible_masked = None  # bool numpy array of shape (84, 84), is_inaccessible
                                            # without the pixels on the four sides set to be accessible forcefully.
        self.reachable_target_idx = None  # int numpy array of (2,), to indicate where the target is in the visual input
        self.reachable_target_size = None  # int, to indicate how many pixels in the target
        self.nearest_inaccessible_idx = None  # int numpy array of (2,), to indicate the nearest inaccessible pixel

        # memory use queues as memory to save the previous observation
        self.visual_hsv_memory = queue.Queue(maxsize=AgentConstants.memory_size)
        self.is_green_memory = queue.Queue(maxsize=AgentConstants.memory_size)
        self.is_brown_memory = queue.Queue(maxsize=AgentConstants.memory_size)
        self.is_red_memory = queue.Queue(maxsize=AgentConstants.memory_size)
        self.is_yellow_memory = queue.Queue(maxsize=AgentConstants.memory_size)
        self.vector_memory = queue.Queue(maxsize=AgentConstants.memory_size)

        # the action that will be returned to the env
        self.current_action = None

        # strategy-related variables
        self.pirouette_step_n = None  # int, counting the how long the agent has remain static and rotating
        self.target_color = None  # the color that the agent is looking for, either "green" or "brown"
        self.exploratory_direction = None  # int, if nowhere to go, go in this direction
        self.not_seeing_target_step_n = None  # int, how long the agent has lost the visual of a target
        self.chase_failed = None  # bool, if it failed to reach a target because path planning failed
        self.search_direction = None  # either left or right, to start searching for the target
        # TODO self.visual_imagery reconstruct mental imagery from primitive perception

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """

        # functional modules
        self.actionStateMachine.reset()
        self.strategy.reset()
        self.perception.reset()
        self.chaser.reset()

        # primitive perception
        self.obs_visual = None
        self.obs_vector = None
        self.obs_visual_hsv = None
        self.done = None
        self.reward = None
        self.info = None
        self.t = t
        self.step_n = 0

        # high-level perceptions
        self.is_green = None
        self.is_brown = None
        self.is_red = None
        self.is_yellow = None
        self.is_blue = None
        self.is_gray = None
        self.target_color = None
        self.is_inaccessible = None
        self.is_inaccessible_masked = None
        self.reachable_target_idx = None
        self.reachable_target_size = None
        self.nearest_inaccessible_idx = None

        # memory
        self.visual_hsv_memory.queue.clear()
        self.is_green_memory.queue.clear()
        self.is_brown_memory.queue.clear()
        self.is_red_memory.queue.clear()
        self.is_yellow_memory.queue.clear()
        self.vector_memory.queue.clear()

        # output action
        self.current_action = None

        # strategy-related variables
        self.pirouette_step_n = 0
        self.target_color = "brown" # start with the non-terminating food
        self.exploratory_direction = None
        self.not_seeing_target_step_n = None
        self.chase_failed = None
        self.search_direction = AgentConstants.left # start with the left to search

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
        :param info: contains auxiliary diagnostic information, including eBrainInfo.
        :return: the action to take, a list or size 2
        """

        # set primitive observations
        self.obs_visual, self.obs_vector = obs
        self.obs_visual_hsv = rgb2hsv(self.obs_visual)
        self.done = done
        self.reward = reward
        self.info = info

        # set high-level observations
        self.perception.perceive()

        # set action by running strategy
        self.strategy.run_strategy()

        return self.current_action
