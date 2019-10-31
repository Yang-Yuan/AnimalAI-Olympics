import numpy as np
import AgentConstants
import warnings
from bresenham import bresenham
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_target_idx = None  # the most recent position of the target
        self.newest_target_size = None  # the most recent size of the target
        self.newest_path = None  # the most recent path to the target
        self.newest_end = None  # the most recent position on the path that can be reached by moving straight (not blocked by obstacles and death zones)
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)  # using A-star to find the path

    def chase(self):
        '''
        This method is for chasing if the target is in view (so the agent knows its position and size)
        :return:
        '''
        self.newest_target_idx = self.agent.reachable_target_idx
        self.newest_target_size = self.agent.reachable_target_size

        self.chase_internal(self.newest_target_idx, self.newest_target_size)

    def chase_in_dark(self):
        '''
        This method is for chasing if the target is not in view, but was in view several step ago.
        When chasing a target, the target will not always be in view. Even though it is not in view,
        it should still be there not far from the agent. So, we use the most recent position of the target
        to keep chasing it, hoping that it will appear again.
        :return:
        '''

        # generate a imaginary target
        imaginary_target_idx, imaginary_target_size = self.imagine_target()

        # this logical branch (for debug purpose) is not possible
        if imaginary_target_idx is None:
            warnings.warn("Can't imagine a target to chase!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.current_action = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        self.chase_internal(imaginary_target_idx, imaginary_target_size)

    def chase_internal(self, target_idx, target_size):
        '''
        Given target_idx and target_size, this method set Agent.current_action
        :param target_idx:
        :param target_size:
        :return:
        '''

        # the path planning is based on the map derived from Agent.is_inaccessible_masked,
        # because the path should consist of only accessible pixels.
        map_matrix = np.logical_not(self.agent.is_inaccessible_masked).astype(np.float)
        # Moreover, the map should be skewed by (similar to) the most recent path to this target.
        # Otherwise, it is possible that the agent might get stuck by infinitely switching
        # between two very different paths without making any progress.
        if self.newest_path is not None:
            map_matrix = self.calculate_path_consistent_matrix(map_matrix)

        # find the path to the target
        grid = Grid(matrix=map_matrix)
        start = grid.node(AgentConstants.standpoint[1], AgentConstants.standpoint[0])  # it accept xy coords.
        end = grid.node(target_idx[1], target_idx[0])  # it accept xy coords.
        path, _ = self.finder.find_path(start, end, grid)

        # if cannot find a path, then declare the failure of chasing and set an empty action
        if path is None or len(path) <= 1:
            warnings.warn("Can't find a path to the target!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.current_action = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        self.newest_path = path
        self.agent.current_action = self.generate_action(path, target_idx, target_size)

    # def is_new_critical_point_in_path(self, line_idx, idx0, idx1):
    #     return 0 <= idx0 < AgentConstants.resolution and 0 <= idx1 < AgentConstants.resolution \
    #            and (not self.agent.is_inaccessible[idx0, idx1]) \
    #            and (not (np.array(line_idx) == [idx0, idx1]).all(axis=1).any())

    def generate_action(self, path, target_idx, target_size):

        # min_col = 0
        # max_col = 83
        min_col = None
        for jj in np.arange(AgentConstants.resolution):
            if not self.agent.is_inaccessible[83, jj]:
                min_col = jj
                break
        if min_col is None:
            warnings.warn("Might have been standing on dangerous area!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.current_action = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        max_col = None
        for jj in np.arange(AgentConstants.resolution)[::-1]:
            if not self.agent.is_inaccessible[83, jj]:
                max_col = jj
                break
        if max_col is None:
            warnings.warn("Might have been standing on dangerous area!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.current_action = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        start = path[0]
        end = path[1]
        for point in path[1:]:
            clear = True
            line_seg_idx = tuple(np.array(list(bresenham(83, min_col, point[1], point[0]))).transpose())
            line_seg_is_inaccessible = self.agent.is_inaccessible[line_seg_idx]
            if line_seg_is_inaccessible.any():
                clear = False
            line_seg_idx = tuple(np.array(list(bresenham(83, max_col, point[1], point[0]))).transpose())
            line_seg_is_inaccessible = self.agent.is_inaccessible[line_seg_idx]
            if line_seg_is_inaccessible.any():
                clear = False

            if clear:
                end = point
            else:
                break

        self.newest_end = end

        if (target_idx[::-1] != end).any():
            target_size = AgentConstants.size_limit

        # self.newest_path = list(bresenham(AgentConstants.standpoint[1], AgentConstants.standpoint[0], end[0], end[1])) \
        #                    + path[path.index(end) + 1:]

        direction_vec = np.array(end) - np.array(start)

        if direction_vec[0] < -AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                return AgentConstants.forward_left
            else:
                return AgentConstants.left
        elif direction_vec[0] > AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                return AgentConstants.forward_right
            else:
                return AgentConstants.right
        else:
            if self.agent.obs_vector[0] < -0.5:
                return AgentConstants.forward_right
            elif self.agent.obs_vector[0] > 0.5:
                return AgentConstants.forward_left
            else:
                if self.agent.obs_vector[2] > 10:
                    return AgentConstants.taxi
                else:
                    return AgentConstants.forward

    def imagine_target(self):
        '''
        This method will find the accessible pixel in the current visual input that is closest to the most recent
        target position in the 2D plance.
        :return:
        '''
        distance = abs(AgentConstants.idx0_grid - np.full((AgentConstants.resolution, AgentConstants.resolution),
                                                          self.newest_target_idx[0])) + \
                   abs(AgentConstants.idx1_grid - np.full((AgentConstants.resolution, AgentConstants.resolution),
                                                          self.newest_target_idx[1]))
        ascending_distance_idx = np.unravel_index(distance.flatten().argsort(),
                                                  (AgentConstants.resolution, AgentConstants.resolution))
        target_idx = None
        for ii, jj in zip(*ascending_distance_idx):
            if not self.agent.is_inaccessible[ii, jj]:
                target_idx = np.array([ii, jj])
                break

        # Since we don't know how large the imaginary target is, simply set its size to one.
        target_size = 1

        return target_idx, target_size

    def calculate_path_consistent_matrix(self, matrix):

        path_idx = np.array(self.newest_path)[:, ::-1]
        for ii in np.arange(AgentConstants.resolution):
            for jj in np.arange(AgentConstants.resolution):
                if matrix[ii, jj] == 1:
                    matrix[ii, jj] = 1 + abs(path_idx - [ii, jj]).sum(axis=1).min() / 2

        return matrix

    def reset(self):
        self.newest_target_idx = None
        self.newest_target_size = None
