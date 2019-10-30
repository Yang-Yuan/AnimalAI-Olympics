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
        self.newest_target_idx = None
        self.newest_target_size = None
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        self.newest_path = None
        self.newest_end = None
        # self.finder = DijkstraFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)

    def chase(self):

        self.newest_target_idx = self.agent.reachable_target_idx
        self.newest_target_size = self.agent.reachable_target_size

        self.chase_internal(self.newest_target_idx, self.newest_target_size)

    def chase_in_dark(self):

        imaginary_target_idx, imaginary_target_size = self.imagine_target()

        if imaginary_target_idx is None:
            warnings.warn("Can't imagine a target to chase!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.current_action = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        self.chase_internal(imaginary_target_idx, imaginary_target_size)

    def chase_internal(self, target_idx, target_size):
        matrix = np.logical_not(self.agent.is_inaccessible_masked).astype(np.float)
        if self.newest_path is not None:
            matrix = self.calculate_path_consistent_matrix(matrix)

        grid = Grid(matrix=matrix)
        start = grid.node(AgentConstants.standpoint[1], AgentConstants.standpoint[0])  # it accept xy coords.
        end = grid.node(target_idx[1], target_idx[0])  # it accept xy coords.
        path, _ = self.finder.find_path(start, end, grid)

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

        # target_idx = AgentConstants.frame_idx[
        #     np.argmin(abs(AgentConstants.frame_idx - self.newest_target_idx).sum(axis=1))]

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
