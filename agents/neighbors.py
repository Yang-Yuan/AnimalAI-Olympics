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
                    if 0 <= iii < resolution and 0 <= jjj < resolution:
                        valid_ii_idx.append(iii)
                        valid_jj_idx.append(jjj)
                neighbor_idx[ii, jj] = (np.array(valid_ii_idx), np.array(valid_jj_idx))
    elif size == 8:
        for ii in range(neighbor_idx.shape[0]):
            for jj in range(neighbor_idx.shape[1]):
                all_ii_idx = [ii, ii - 1, ii + 1, ii, ii, ii - 1, ii + 1, ii - 1, ii + 1]
                all_jj_idx = [jj, jj, jj, jj - 1, jj + 1, jj - 1, jj + 1, jj + 1, jj - 1]
                valid_ii_idx = []
                valid_jj_idx = []
                for iii, jjj in zip(all_ii_idx, all_jj_idx):
                    if 0 <= iii < resolution and 0 <= jjj < resolution:
                        valid_ii_idx.append(iii)
                        valid_jj_idx.append(jjj)
                neighbor_idx[ii, jj] = (np.array(valid_ii_idx), np.array(valid_jj_idx))
    elif size == 12:
        for ii in range(neighbor_idx.shape[0]):
            for jj in range(neighbor_idx.shape[1]):
                all_ii_idx = [ii, ii - 1, ii + 1, ii, ii, ii - 1, ii + 1, ii - 1, ii + 1, ii - 2, ii + 2, ii, ii]
                all_jj_idx = [jj, jj, jj, jj - 1, jj + 1, jj - 1, jj + 1, jj + 1, jj - 1, jj, jj, jj - 2, jj + 2]
                valid_ii_idx = []
                valid_jj_idx = []
                for iii, jjj in zip(all_ii_idx, all_jj_idx):
                    if 0 <= iii < resolution and 0 <= jjj < resolution:
                        valid_ii_idx.append(iii)
                        valid_jj_idx.append(jjj)
                neighbor_idx[ii, jj] = (np.array(valid_ii_idx), np.array(valid_jj_idx))
    else:
        return None

    return neighbor_idx


def sparseOrthogonalNeighbors(radius):
    neighbor_idx = np.empty((resolution, resolution), dtype=tuple)

    orthogonalDirections = [[-1, 1, 0, 0, -1, -1, 1, 1], [0, 0, -1, 1, -1, 1, -1, 1]]

    for ii in range(neighbor_idx.shape[0]):
        for jj in range(neighbor_idx.shape[1]):
            all_ii_idx = [ii]
            all_jj_idx = [jj]

            for delta_i, delta_j in zip(orthogonalDirections[0], orthogonalDirections[1]):
                for delta_r in range(1, radius + 1):
                    all_ii_idx.append(ii + delta_i * delta_r)
                    all_jj_idx.append(jj + delta_j * delta_r)

            valid_ii_idx = []
            valid_jj_idx = []
            for iii, jjj in zip(all_ii_idx, all_jj_idx):
                if 0 <= iii < resolution and 0 <= jjj < resolution:
                    valid_ii_idx.append(iii)
                    valid_jj_idx.append(jjj)
            neighbor_idx[ii, jj] = (np.array(valid_ii_idx), np.array(valid_jj_idx))

    return neighbor_idx
