
import numpy as np

resolution = 84


def orthogonalNeighbors(size):
    neighbor_idx = np.empty((resolution, resolution), dtype=tuple)

    if size == 4:
        for ii in range(neighbor_idx.shape[0]):
            for jj in range(neighbor_idx.shape[1]):
                all_ii_idx = [ii, ii - 1, ii + 1, ii, ii]
                all_jj_idx = [jj, jj, jj, jj - 1, jj + 1]
                valid_ii_idx = []
                valid_jj_idx = []
                for iii, jjj in zip(all_ii_idx, all_jj_idx):
                    if 0 <= iii and iii < resolution and 0 <= jjj and jjj < resolution:
                        valid_ii_idx.append(iii)
                        valid_jj_idx.append(jjj)
                neighbor_idx[ii, jj] = (valid_ii_idx, valid_jj_idx)
    elif size == 8:
        for ii in range(neighbor_idx.shape[0]):
            for jj in range(neighbor_idx.shape[1]):
                all_ii_idx = [ii, ii - 1, ii + 1, ii, ii, ii - 1, ii + 1, ii - 1, ii + 1]
                all_jj_idx = [jj, jj, jj, jj - 1, jj + 1, jj - 1, jj + 1, jj + 1, jj - 1]
                valid_ii_idx = []
                valid_jj_idx = []
                for iii, jjj in zip(all_ii_idx, all_jj_idx):
                    if 0 <= iii and iii < resolution and 0 <= jjj and jjj < resolution:
                        valid_ii_idx.append(iii)
                        valid_jj_idx.append(jjj)
                neighbor_idx[ii, jj] = (valid_ii_idx, valid_jj_idx)
    else:
        return None

    return neighbor_idx

