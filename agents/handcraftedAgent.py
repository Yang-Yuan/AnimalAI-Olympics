import numpy as np
from skimage import measure


class Agent(object):
    green = [0.506, 0.749, 0.255]
    predfined_colors_h = {"green": 0.24865047}

    color_diff_limit = 0.075
    position_diff_limit = 1
    size_limit = 5

    center_of_view = [41.5, 41.5]
    aim_error_limit = 5
    hl = 2
    default_test_length = 1000
    resolution = 84

    bin_size_limit = 10

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """
        self.t = 0
        self.step_n = 0
        self.total_reward = 0
        self.pirouette_step_n = 0
        self.diff_center_old = None
        self.target_size_old = None
        self.visual_memory = None

        self.n_bins = 30
        self.bin_edges = np.linspace(start=0, stop=1, num=self.n_bins + 1)
        self.bin_length = 1 / self.n_bins
        self.bin_sizes = np.zeros(self.n_bins, dtype = np.int)
        self.bin_colors = np.zeros(self.n_bins)
        self.predefined_colors_bins = {"green": np.digitize(Agent.predfined_colors_h.get("green"), self.bin_edges)}

        self.visual_imagery = np.zeros((Agent.resolution, Agent.resolution))
        self.bin_pixels_idx = []


    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.t = t
        self.step_n = 0
        self.total_reward = 0
        self.pirouette_step_n = 0
        self.visual_memory = np.zeros((self.t, Agent.resolution, Agent.resolution), dtype=np.int)

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
        if done:
            return [0, 0]

        # Jaja, magic code!
        if self.pirouette_step_n > 70:
            self.pirouette_step_n = 0
            return [1, 0]

        obs_visual, obs_vector = obs
        obs_visual_h = Agent.toHue(obs_visual)

        # update bin_labels, self.bin_sizes, self.bin_color
        bin_labels = np.digitize(obs_visual_h, self.bin_edges)
        self.bin_pixels_idx = []
        for bin_label in range(1, self.n_bins + 1):
            self.bin_pixels_idx.append(np.array(np.where(bin_labels == bin_label)))
            self.bin_sizes[bin_label - 1] = self.bin_pixels_idx[bin_label - 1].shape[1]
            if 0 != self.bin_sizes[bin_label - 1]:
                self.bin_colors[bin_label - 1] = obs_visual_h[tuple(self.bin_pixels_idx[bin_label - 1])].mean(axis=0)

        # rectify bins so that the pixels in each bin are not only close in color but also constitute one or more
        # connected areas
        # In other words, I don't allow these boundary points to be a single bin;
        # they must affiliate to some object-level bins.
        # for bin_idx in range(self.n_bins):
        #     if self.bin_sizes[bin_idx] > Agent.bin_size_limit \
        #             or (bin_idx in self.predefined_colors_bins.values()):
        #         continue
        #
        #     # fine the nearest significant bin
        #     # and add this bin into the nearest significant bin
        #     delta = 1
        #     while True:
        #         if bin_idx - delta >= 0 and self.bin_sizes[bin_idx - delta] > Agent.bin_size_limit:
        #             self.bin_sizes[bin_idx - delta] += self.bin_sizes[bin_idx]
        #             self.bin_sizes[bin_idx] = 0
        #             self.bin_pixels_idx[bin_idx - delta] = np.concatenate( \
        #                 (self.bin_pixels_idx[bin_idx], self.bin_pixels_idx[bin_idx - delta]), axis = 1)
        #             break
        #
        #         if bin_idx + delta < self.n_bins and self.bin_sizes[bin_idx + delta] > Agent.bin_size_limit:
        #             self.bin_sizes[bin_idx + delta] += self.bin_sizes[bin_idx]
        #             self.bin_sizes[bin_idx] = 0
        #             self.bin_pixels_idx[bin_idx + delta] = np.concatenate( \
        #                 (self.bin_pixels_idx[bin_idx], self.bin_pixels_idx[bin_idx + delta]), axis = 1)
        #             break
        #
        #         delta += 1

        # This part will be uncommented after each bin has been optimized (purified)
        # # try to build an visual imagery reconstruct the image using the rectified bin
        # for bin_idx in range(self.n_bins):
        #     self.visual_imagery[tuple(self.bin_pixels_idx[bin_idx])] = self.bin_colors[bin_idx]
        # # for debug
        # self.visual_imagery = -np.ones((Agent.resolution, Agent.resolution))
        # if (self.visual_imagery == -1).any():
        #     raise Exception("You missed some pixel, idiot!")

        diff_green = abs(obs_visual_h - Agent.predfined_colors_h.get("green"))
        is_green = diff_green < Agent.color_diff_limit


        # self.visual_memory[self.step_n] = is_green
        self.step_n += 1

        # print(obs_vector)
        # For debug
        # self.total_reward += reward
        # print("step:{} reward:{} total_reward:{} done:{}".format(self.step_n, reward, self.total_reward, done))
        # if 250 == self.step_n:
        #     print(diff_green.min())
        #     print("Failed")
        #     sys.exit(1)

        if not is_green.any():
            self.pirouette_step_n += 1
            return [0, 1]

        if 1 == is_green.sum():
            diff_center = np.array(np.where(is_green)).transpose()[0] - Agent.center_of_view
            target_size = 1
        else:
            labels, label_num = measure.label(input=is_green, background=False, return_num=True, connectivity=1)
            sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
            target_label = np.argmax(sizes) + 1
            center_of_target = np.array(np.where(labels == target_label)).mean(axis=1)
            diff_center = center_of_target - Agent.center_of_view
            target_size = sizes[target_label - 1]

        # diff_center_old
        # diff_center
        # target_size_old
        # target_size
        # obs_vector

        if diff_center[1] < -Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            if target_size < Agent.size_limit:
                self.pirouette_step_n = 0
                return [1, 2]
            else:
                self.pirouette_step_n += 1
                return [0, 2]
        elif diff_center[1] > Agent.aim_error_limit * (1 + np.exp(-target_size / Agent.hl)):
            if target_size < Agent.size_limit:
                self.pirouette_step_n = 0
                return [1, 1]
            else:
                self.pirouette_step_n += 1
                return [0, 1]
        else:
            self.pirouette_step_n = 0
            return [1, 0]

    @staticmethod
    def toHue(rgb):
        out_h = np.zeros(rgb.shape[:-1])

        out_v = rgb.max(-1) # max
        delta = rgb.ptp(-1) # max - min

        old_settings = np.seterr(invalid='ignore')

        # red is max
        idx = (rgb[:, :, 0] == out_v)
        out_h[idx] = (rgb[idx, 1] - rgb[idx, 2]) / delta[idx]

        # green is max
        idx = (rgb[:, :, 1] == out_v)
        out_h[idx] = 2. + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]

        # blue is max
        idx = (rgb[:, :, 2] == out_v)
        out_h[idx] = 4. + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]

        # normorlization
        out_h = (out_h / 6.) % 1.
        out_h[delta == 0.] = 0.

        np.seterr(**old_settings)

        # remove NaN
        out_h[np.isnan(out_h)] = 0

        return out_h
