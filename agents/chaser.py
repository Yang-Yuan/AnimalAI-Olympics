import numpy as np
import AgentConstants
import warnings
import agentUtils
from bresenham import bresenham
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
# from pathfinding.finder.dijkstra import DijkstraFinder


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_target_idx = None
        self.newest_target_size = None
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        # self.finder = DijkstraFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)

    def chase(self):

        self.newest_target_idx = self.agent.reachable_target_idx
        self.newest_target_size = self.agent.reachable_target_size

        self.chase_internal(self.newest_target_idx, self.newest_target_size)

    def chase_in_dark(self):

        imaginary_target_idx, imaginary_target_size = self.imagine_target()

        self.chase_internal(imaginary_target_idx, imaginary_target_size)

    def chase_internal(self, target_idx, target_size):

        grid = Grid(matrix = np.logical_not(self.agent.is_inaccessible))
        start = grid.node(AgentConstants.standpoint[1], AgentConstants.standpoint[0])
        end = grid.node(target_idx[1], target_idx[0])
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
               and (not (np.array(line_idx) == [idx0, idx1]).all(axis = 1).any())

    def generate_action(self, critical_points, target_idx, target_size):

        start = critical_points[0]
        end = None
        for point in critical_points[1:]:
            line_seg_idx = tuple(np.array(list(bresenham(start[0], start[1],
                                                         point[0], point[1]))).transpose())
            line_seg_is_inaccessible = self.agent.is_inaccessible[line_seg_idx]
            if line_seg_is_inaccessible.any():
                break
            else:
                end = point

        if (target_idx != end).any():
            target_size = AgentConstants.size_limit

        direction_vec = np.array(end) - np.array(start)

        if direction_vec[1] < -AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                return AgentConstants.forward_left
            else:
                return AgentConstants.left
        elif direction_vec[1] > AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
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

    def reset(self):
        self.newest_target_idx = None
        self.newest_target_size = None
