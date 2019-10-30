import AgentConstants
import numpy as np
from skimage import measure
import agentUtils


class Perception(object):

    def __init__(self, agent):
        self.agent = agent

    def perceive(self):
        '''
        This method will recognize colors, update targets and save memory.
        :return:
        '''

        # we limit the size of memory, so before saving new observations,
        # delete the old ones if the memory is full
        if self.agent.is_green_memory.full():
            self.agent.is_green_memory.get()
        if self.agent.is_brown_memory.full():
            self.agent.is_brown_memory.get()
        if self.agent.is_red_memory.full():
            self.agent.is_red_memory.get()
        if self.agent.vector_memory.full():
            self.agent.vector_memory.get()

        # label each pixel to see if they are green(food), brown(also food) , red(danger), gray(walls) or blue(sky).
        self.agent.is_green = abs(self.agent.obs_visual_hsv[:, :, 0] - AgentConstants.predefined_colors_h.get("green")[0]) < AgentConstants.green_tolerance
        self.agent.is_brown = abs(self.agent.obs_visual - AgentConstants.predefined_colors.get("brown")).max(axis=2) < AgentConstants.brown_tolerance
        self.agent.is_red = (abs(self.agent.obs_visual_hsv - AgentConstants.predefined_colors_h.get("red")) < AgentConstants.red_tolerance).all(axis=2)
        self.agent.is_gray = np.logical_and(abs(self.agent.obs_visual[:, :, 0] - self.agent.obs_visual[:, :, 1]) < 0.001, abs(self.agent.obs_visual[:, :, 1] - self.agent.obs_visual[:, :, 2]) < 0.001)
        self.agent.is_blue = abs(self.agent.obs_visual - AgentConstants.predefined_colors.get("sky_blue")).max(axis=2) < AgentConstants.sky_blue_tolerance

        # filter out some noisy pixel labels
        self.agent.is_brown = self.agent.is_brown if agentUtils.is_color_significant(self.agent.is_brown, AgentConstants.brown_size_limit) else AgentConstants.all_false
        self.agent.is_red = self.agent.is_red if agentUtils.is_color_significant(self.agent.is_red, AgentConstants.red_size_limit) else AgentConstants.all_false
        self.agent.is_gray = self.agent.is_gray if agentUtils.is_color_significant(self.agent.is_gray, AgentConstants.gray_size_limit) else AgentConstants.all_false

        # enlarge the area of dangerous color so that the agent is less likely to kill itself.
        self.agent.is_red = self.puff_red(delta=3)

        # merge red, blue and gray to generate the inaccessible area for path planning later.
        self.synthesize_is_inaccessible()

        # if food exists, update this position and size of it for path planning
        self.update_target()

        # update the closest inaccessible pixel for the safety of exploration.
        self.update_nearest_inaccessible_idx()

        # save memory
        self.agent.is_green_memory.put(self.agent.is_green)
        self.agent.is_brown_memory.put(self.agent.is_brown)
        self.agent.is_red_memory.put(self.agent.is_red)
        self.agent.vector_memory.put(self.agent.obs_vector[0])

    def renew_target_from_panorama(self):

        if self.agent.pirouette_step_n == AgentConstants.pirouette_step_limit:

            green_memory = np.array(self.agent.is_green_memory.queue)[-AgentConstants.pirouette_step_limit:]
            if green_memory.any():
                self.agent.target_color = "green"
                best_direction = green_memory.sum(axis=(1, 2)).argmax()
                if 0 <= best_direction < 30:
                    if self.agent.search_direction == AgentConstants.left:
                        self.agent.search_direction = AgentConstants.left
                    else:
                        self.agent.search_direction = AgentConstants.right
                else:
                    if self.agent.search_direction == AgentConstants.left:
                        self.agent.search_direction = AgentConstants.right
                    else:
                        self.agent.search_direction = AgentConstants.left
                self.agent.exploratory_direction = None

            else:
                self.agent.exploratory_direction = np.random.choice(AgentConstants.pirouette_step_limit)
                self.agent.target_color = "brown"

    def renew_target(self):

        if self.agent.target_color == "green" and self.agent.is_brown.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "brown"
                return True

        if self.agent.target_color is None and self.agent.is_green.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_green)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "green"
                return True

        if self.agent.target_color is None and self.agent.is_brown.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            if self.agent.reachable_target_idx is None:
                return False
            else:
                self.agent.target_color = "brown"
                return True

        return False

    def is_front_safe(self):
        if self.agent.nearest_inaccessible_idx is None:
            return True
        else:
            return (AgentConstants.resolution - self.agent.nearest_inaccessible_idx[0]) > \
                   AgentConstants.minimal_dist_to_in_accessible

    def is_static(self):
        return (self.agent.obs_vector == 0).all()

    def is_nearly_static(self):
        return (abs(self.agent.obs_vector) < 0.1).all()

    def is_found(self):
        return self.agent.reachable_target_idx is not None

    def is_chasing_done(self):
        return self.agent.reward is not None and self.agent.reward > 0

    def synthesize_is_inaccessible(self):
        # TODO maybe add the more inaccessible things here, but...
        self.agent.is_inaccessible = np.logical_or(np.logical_or(self.agent.is_gray,
                                                                 self.agent.is_red),
                                                                 self.agent.is_blue)
        self.agent.is_inaccessible_masked = np.logical_and(self.agent.is_inaccessible,
                                                           np.logical_not(AgentConstants.frame_mask))

    def find_reachable_target(self, is_color):
        if is_color.any():
            labels, label_num = measure.label(input=is_color,
                                              background=False,
                                              return_num=True,
                                              connectivity=1)
            sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
            for ii in np.argsort(sizes)[::-1]:
                label = ii + 1
                idx = np.argwhere(labels == label)
                idx_idx = idx.argmax(axis=0)[0]
                lowest_idx = idx[idx_idx]
                return lowest_idx, sizes[ii]
            #     if not self.agent.is_inaccessible[lowest_idx[0]: lowest_idx[0] + 3, lowest_idx[1]].any():
            #         return lowest_idx, sizes[ii]
            #
            # return None, None
        else:
            return None, None

    def update_target(self):

        if self.agent.is_brown.any():
            self.agent.target_color = "brown"
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_brown)
            return
        elif self.agent.target_color == "green" and self.agent.is_green.any():
            self.agent.reachable_target_idx, self.agent.reachable_target_size = self.find_reachable_target(
                self.agent.is_green)
            return
        else:
            self.agent.reachable_target_idx = None
            self.agent.reachable_target_size = None

    def update_nearest_inaccessible_idx(self):
        idx = np.argwhere(np.logical_and(AgentConstants.road_mask, self.agent.is_red))
        if 0 == len(idx):
            self.agent.nearest_inaccessible_idx = None
        else:
            self.agent.nearest_inaccessible_idx = idx[idx[:, 0].argmax()]

    def puff_red(self, delta):
        new_is_red = self.agent.is_red.copy()
        shifted_up = self.agent.is_red.copy()
        shifted_down = self.agent.is_red.copy()
        shifted_left = self.agent.is_red.copy()
        shifted_right = self.agent.is_red.copy()
        for _ in np.arange(delta):
            shifted_up = np.roll(shifted_up, (-1, 0), (0, 1))
            shifted_up[-1, :] = False
            new_is_red = np.logical_or(new_is_red, shifted_up)
            shifted_down = np.roll(shifted_down, (1, 0), (0, 1))
            shifted_down[0, :] = False
            new_is_red = np.logical_or(new_is_red, shifted_down)
            shifted_left = np.roll(shifted_left, (0, -1), (0, 1))
            shifted_left[:, -1] = False
            new_is_red = np.logical_or(new_is_red, shifted_left)
            shifted_right = np.roll(shifted_right, (0, 1), (0, 1))
            shifted_right[:, 0] = False
            new_is_red = np.logical_or(new_is_red, shifted_right)
        new_is_red = np.logical_and(np.logical_and(new_is_red,
                                                   np.logical_not(self.agent.is_green)),
                                    np.logical_not(self.agent.is_brown))
        return new_is_red

    def reset(self):
        pass
