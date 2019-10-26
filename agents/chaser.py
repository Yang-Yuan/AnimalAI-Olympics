import numpy as np
import AgentConstants
import warnings
import agentUtils
from bresenham import bresenham
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
# from pathfinding.finder.dijkstra import DijkstraFinder


# from pathfinding.finder.dijkstra import DijkstraFinder


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_target_idx = None
        self.newest_target_size = None
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        self.newest_end_point = None
        # self.finder = DijkstraFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)

    def chase(self):

        self.newest_target_idx = self.agent.reachable_target_idx
        self.newest_target_size = self.agent.reachable_target_size

        self.chase_internal(self.newest_target_idx, self.newest_target_size)

    def chase_in_dark(self):

        imaginary_target_idx, imaginary_target_size = self.imagine_target()

        self.chase_internal(imaginary_target_idx, imaginary_target_size)

    def chase_internal(self, target_idx, target_size):
        matrix = np.logical_not(self.agent.is_inaccessible).astype(np.float)
        if self.newest_end_point is not None:
            matrix *= self.calculate_path_consistent_gain_ratio()

        grid = Grid(matrix=matrix)
        start = grid.node(AgentConstants.standpoint[1], AgentConstants.standpoint[0])  # it accept xy coords.
        end = grid.node(target_idx[1], target_idx[0])  # it accept xy coords.
        path, _ = self.finder.find_path(start, end, grid)

        if path is None or 0 == len(path):
            warnings.warn("Can't find a path to the target!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.agent.currentAction = AgentConstants.taxi
            self.agent.chase_failed = True
            return

        self.agent.currentAction = self.generate_action(path, target_idx, target_size)

    def is_new_critical_point_in_path(self, line_idx, idx0, idx1):
        return 0 <= idx0 < AgentConstants.resolution and 0 <= idx1 < AgentConstants.resolution \
               and (not self.agent.is_inaccessible[idx0, idx1]) \
               and (not (np.array(line_idx) == [idx0, idx1]).all(axis=1).any())

    def generate_action(self, path, target_idx, target_size):

        start = path[0]
        end = None
        for point in path[1:]:
            line_seg_idx = tuple(np.array(list(bresenham(start[1], start[0],
                                                         point[1], point[0]))).transpose())
            line_seg_is_inaccessible = self.agent.is_inaccessible[line_seg_idx]
            if line_seg_is_inaccessible.any():
                break
            else:
                end = point

        self.newest_end_point = end
        print(end)

        if (target_idx[::-1] != end).any():
            target_size = AgentConstants.size_limit

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
            return AgentConstants.forward

    def imagine_target(self):

        target_idx = AgentConstants.frame_idx[
            np.argmin(abs(AgentConstants.frame_idx - self.newest_target_idx).sum(axis=1))]
        target_size = 1

        return target_idx, target_size

    def calculate_path_consistent_gain_ratio(self):
        end_idx0 = np.full((AgentConstants.resolution, AgentConstants.resolution), self.newest_end_point[1])
        end_idx1 = np.full((AgentConstants.resolution, AgentConstants.resolution), self.newest_end_point[0])
        idx0, idx1 = np.meshgrid(np.arange(AgentConstants.resolution), np.arange(AgentConstants.resolution),
                                 indexing="ij")
        ratio = np.exp2(abs(idx0 - end_idx0) + abs(idx1 - end_idx1))
        return ratio

    def reset(self):
        self.newest_target_idx = None
        self.newest_target_size = None
