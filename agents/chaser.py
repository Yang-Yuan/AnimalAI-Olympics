import numpy as np
import AgentConstants
import warnings
import agentUtils
from bresenham import bresenham


class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_target_idx = None
        self.newest_target_size = None

    def chase(self):

        self.newest_target_idx = self.agent.reachable_target_idx
        self.newest_target_size = self.agent.reachable_target_size

        self.chase_internal(self.newest_target_idx, self.newest_target_size)

    def chase_in_dark(self):

        imaginary_target_idx, imaginary_target_size = self.imagine_target()

        self.chase_internal(imaginary_target_idx, imaginary_target_size)

    def chase_internal(self, target_idx, target_size):

        critical_points_in_path = [AgentConstants.standpoint, target_idx]

        line_idx = agentUtils.render_line_segments(critical_points_in_path)

        line_is_inaccessible = self.agent.is_inaccessible[tuple(np.array(line_idx).transpose())]

        while line_is_inaccessible.any():

            idx_idx = np.argwhere(line_is_inaccessible).flatten()

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
                warnings.warn("Chase Failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.agent.currentAction = AgentConstants.taxi
                self.agent.chase_failed = True

        self.agent.currentAction = self.generate_action(critical_points_in_path, target_idx, target_size)

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

        direction_vec = end - start

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

        target_center = AgentConstants.frame_idx[
            np.argmin(abs(AgentConstants.frame_idx - self.newest_target_center).sum(axis=1))]
        target_size = 1

        return target_center, target_size

    def reset(self):
        self.newest_target_idx = None
        self.newest_target_size = None
