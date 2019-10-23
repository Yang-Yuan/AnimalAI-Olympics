from skimage import measure
import numpy as np
import AgentConstants
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import warnings
import agentUtils
from bresenham import bresenham


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_is_color = None

    def chase(self):
        self.newest_is_color = self.agent.is_color
        self.chase_internal(self.newest_is_color)

    def chase_in_dark(self):
        imaginary_is_color = self.newest_is_color  # TODO enhance
        self.chase_internal(imaginary_is_color)

    def chase_internal(self, is_color):

        labels, label_num = measure.label(input=is_color, background=False, return_num=True, connectivity=1)
        sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
        target_label = np.argmax(sizes) + 1
        target_center = np.array(np.where(labels == target_label)).mean(axis=1).astype(np.int)
        target_size = sizes[target_label - 1]

        # grid = Grid(np.logical_not(self.agent.is_red))
        # start = grid.node(AgentConstants.standpoint[0], AgentConstants.standpoint[1])
        # end = grid.node(target_center[0], target_center[1])
        # finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        # path, runs = finder.find_path(start, end, grid)

        critical_points_in_path = [AgentConstants.standpoint, target_center]

        line_idx = agentUtils.render_line_segments(critical_points_in_path)

        line_is_red = self.agent.is_red[tuple(line_idx.transpose())]
        while not line_is_red.any():

            idx_idx = np.argwhere(line_is_red).flatten()

            for ii in idx_idx:

                red_pixel_idx = line_idx[ii]
                new_idx = None
                delta = 1
                while True:
                    if (not self.agent.is_red[red_pixel_idx[0] + delta, red_pixel_idx[1]]) \
                            and (line_idx == [red_pixel_idx[0] + delta, red_pixel_idx[1]]).sum(axis=1).any():
                        new_idx = [red_pixel_idx[0] + delta, red_pixel_idx[1]]
                        break
                    if (not self.agent.is_red[red_pixel_idx[0] - delta, red_pixel_idx[1]]) \
                            and (line_idx == [red_pixel_idx[0] - delta, red_pixel_idx[1]]).sum(axis=1).any():
                        new_idx = [red_pixel_idx[0] - delta, red_pixel_idx[1]]
                        break
                    if (not self.agent.is_red[red_pixel_idx[0], red_pixel_idx[1] + delta]) \
                            and (line_idx == [red_pixel_idx[0], red_pixel_idx[1] + delta]).sum(axis=1).any():
                        new_idx = [red_pixel_idx[0], red_pixel_idx[1] + delta]
                        break
                    if (not self.agent.is_red[red_pixel_idx[0], red_pixel_idx[1] - delta]) \
                            and (line_idx == [red_pixel_idx[0], red_pixel_idx[1] - delta]).sum(axis=1).any():
                        new_idx = [red_pixel_idx[0], red_pixel_idx[1] - delta]
                        break

                if new_idx is not None:
                    break

            if new_idx is not None:
                for jj in np.arange(start=ii, end=len(line_idx)):
                    insert_idx = None
                    try:
                        insert_idx = critical_points_in_path.index(line_idx[jj])
                    except ValueError:
                        pass
                    if insert_idx is not None:
                        critical_points_in_path.insert(insert_idx, new_idx)
                        line_idx = agentUtils.render_line_segments(critical_points_in_path)
            else:
                warnings.warn("Cannot find a path to the target.")
                # TODO

        self.generate_action(critical_points_in_path)

    def generate_action(self, critical_points, target_center, target_size):

        start = critical_points[0]
        end = None
        for point in critical_points[1:]:
            line_seg_idx = tuple(np.array(list(bresenham(start[0], start[1],
                                                         point[0], point[1]))).transpose())
            line_seg_is_red = self.agent.is_red[line_seg_idx]
            if line_seg_is_red.any():
                end = point
            else:
                break

        vec = np.array(end) - np.array(end)



        if diff_center[1] < -AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                self.pirouette_step_n = 0
                return [1, 2]
            else:
                self.pirouette_step_n += 1
                return [0, 2]
        elif diff_center[1] > AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                self.pirouette_step_n = 0
                return [1, 1]
            else:
                self.pirouette_step_n += 1
                return [0, 1]
        else:
            self.pirouette_step_n = 0
            return [1, 0]
