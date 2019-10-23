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
        self.newest_target_center = None
        self.newest_target_size = None

    def chase(self):

        self.newest_target_center, self.newest_target_size = self.find_reachable_object()

        if self.newest_target_center is None:
            return

        self.chase_internal(self.newest_target_center, self.newest_target_size)

    def chase_in_dark(self):
        imaginary_target_center, imaginary_target_size = self.imagine_chasable_object()

        self.chase_internal(imaginary_target_center, imaginary_target_size)

    def chase_internal(self, target_center, target_size):

        critical_points_in_path = [AgentConstants.standpoint, target_center]

        line_idx = agentUtils.render_line_segments(critical_points_in_path)

        line_is_red = self.agent.is_red[tuple(line_idx.transpose())]

        while not line_is_red.any():

            idx_idx = np.argwhere(line_is_red).flatten()

            new_idx = None
            for ii in idx_idx:

                red_pixel_idx = line_idx[ii]
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
                return False

        self.agent.currentAction = self.generate_action(critical_points_in_path, target_center, target_size)
        return True

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

        if target_center != end:
            target_size = AgentConstants.size_limit

        direction_vec = np.array(end) - np.array(start)

        if direction_vec[1] < -AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                return AgentConstants.forward_left
            else:
                return AgentConstants.left
        elif direction_vec[1] > AgentConstants.aim_error_limit * (1 + np.exp(-target_size / AgentConstants.hl)):
            if target_size < AgentConstants.size_limit:
                return AgentConstants.right
            else:
                return AgentConstants.right
        else:
            return AgentConstants.forward

    def synthesize_is_inaccessible(self):
        # TODO maybe add the walls here, but...
        is_inaccessible = np.copy(self.agent.is_red)
        is_inaccessible = np.logical_and(is_inaccessible, np.logical_not(AgentConstants.frame_mask))
        return is_inaccessible

    def find_reachable_object(self):
        labels, label_num = measure.label(input=self.agent.is_color, background=False, return_num=True, connectivity=1)
        sizes = [(labels == label).sum() for label in range(1, label_num + 1)]

        target_center = None
        for ii in np.argsort(sizes)[::-1]:
            label = ii + 1
            idx = np.argwhere(labels = label)
            idx_idx = idx.argmax(axis = 0)[1]
            lowest_idx = idx[idx_idx]


        target_label = np.argmax(sizes) + 1
        target_center = np.array(np.where(labels == target_label)).mean(axis=1).astype(np.int)
        target_size = sizes[target_label - 1]

        return target_center, target_size

    def imagine_chasable_object(self):

        target_center = AgentConstants.frame_idx[
            np.argmin(abs(AgentConstants.frame_idx - self.newest_target_center).sum(axis=1))]
        target_size = 1

        return target_center, target_size
