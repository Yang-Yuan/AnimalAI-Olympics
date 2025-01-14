import numpy as np

resolution = 84

# TODO these two parameters need to be tuned
bin_size_limit = 10
gradient_limit = 0.03

n_bins = 20
bin_edges = np.linspace(start=0, stop=1, num=n_bins + 1)
bin_length = 1 / n_bins
bin_centers = (bin_edges[: -1] + bin_edges[1:]) / 2


def bispaceClustering(visual, cluster_centers):
    old_visual = visual
    old_cluster_centers = cluster_centers
    p_c4xy = computePc4xy(old_visual)
    p_k4c, p_c4k = computePk4cAndPc4k(old_cluster_centers)
    while True:
        # Agent.image1.set_data(plt.cm.hsv(old_visual))
        # Agent.fig.canvas.draw()
        # Agent.fig.canvas.flush_events()

        p_k4xy = np.tensordot(p_k4c, p_c4xy, axes=1)
        p_c4xy = np.tensordot(p_c4k, p_k4xy, axes=1)

        # new_visual = np.tensordot(Agent.bin_centers, p_c4xy, axes = 1)
        # new_visual = Agent.bin_centers[p_c4xy.argmax(axis = 0)]
        new_visual = Agent.sampleNewVisual(p_c4xy)
        new_cluster_centers = np.tensordot(p_k4xy, new_visual, axes=2) / p_k4xy.sum(axis=(1, 2))
        # TODO there might be some numerical error here that new_cluster_centers wiil go beyond [0, 1]

        if canStop(old_visual, old_cluster_centers, new_visual, new_cluster_centers):
            break
        else:
            old_visual = new_visual
            old_cluster_centers = new_cluster_centers

        p_c4xy = computePc4xy(old_visual)
        p_k4c, p_c4k = computePk4cAndPc4k(old_cluster_centers)

    return new_visual


def sampleNewVisual(p_c4xy):
    new_visual = np.zeros((Agent.resolution, Agent.resolution), dtype=float)
    for ii in range(p_c4xy.shape[1]):
        for jj in range(p_c4xy.shape[2]):
            # new_visual[ii, jj] = np.random.choice(Agent.bin_centers, p = p_c4xy[:, ii, jj])
            new_visual[ii, jj] = Agent.bin_centers[p_c4xy[:, ii, jj].argmax()]
    return new_visual


def restartNewVisual(p_c4xy):
    new_visual = np.zeros((Agent.resolution, Agent.resolution), dtype=float)
    for ii in range(p_c4xy.shape[1]):
        for jj in range(p_c4xy.shape[2]):
            new_visual[ii, jj] = np.random.choice(Agent.bin_centers, p=p_c4xy[:, ii, jj])
    return new_visual


def computePc4xy(visual):
    p_c4xy = np.zeros(Agent.bin_centers.shape + visual.shape, dtype=float)
    for ii in range(visual.shape[0]):
        for jj in range(visual.shape[1]):
            p_c4xy[:, ii, jj], _ = np.histogram(visual[Agent.twel_neighbor_idx[ii, jj]],
                                                bins=Agent.bin_edges)
            p_c4xy[:, ii, jj] = p_c4xy[:, ii, jj] / p_c4xy[:, ii, jj].sum()
    return p_c4xy


def computePc4xy_old(visual):
    # this way:
    # [defining neighborhood and computing the mean color
    # of the neighborhood and then computing the probability of each color
    # according to the distance between the bin_colors and mean color]
    # might be wrong.
    # because p_c4xy represents the clustering in the coordinate space
    # (while p_k4xy represents clustering in the colors space),
    # the distance used to determine the probability of each bin_color
    # should be measured in the coordinate space
    # (while the distance for p_k4xy should be measured in color space).

    neighbor_idx = truncatedMinimalNeighbors(visual, Agent.gradient_limit)
    neighbor_mean_visual = calculateMeanVisual(visual, neighbor_idx)

    cluster_colors_visual = np.empty(Agent.bin_centers.shape + visual.shape, dtype=float)
    for ii in range(len(Agent.bin_centers)):
        cluster_colors_visual[ii] = np.full(neighbor_mean_visual.shape, Agent.bin_centers[ii])

    cluster_colors_diffs = abs(cluster_colors_visual - neighbor_mean_visual)

    old_settings = np.seterr(invalid='ignore')
    p_c4xy = np.exp(1 / cluster_colors_diffs)
    p_c4xy = p_c4xy / p_c4xy.sum(axis=0)
    p_c4xy[np.isnan(p_c4xy)] = 1
    np.seterr(**old_settings)

    # p_c4xy = np.exp(-cluster_colors_diffs)
    # p_c4xy = p_c4xy / p_c4xy.sum(axis=0)

    return p_c4xy


def computePk4cAndPc4k(cluster_centers):
    cluster_colors_spectrum = np.empty(cluster_centers.shape + Agent.bin_centers.shape, dtype=float)
    for ii in range(len(cluster_centers)):
        cluster_colors_spectrum[ii] = np.full(Agent.bin_centers.shape, cluster_centers[ii])

    cluster_colors_diffs = abs(cluster_colors_spectrum - Agent.bin_centers)

    old_settings = np.seterr(invalid='ignore')
    p_k4c = np.exp(1 / cluster_colors_diffs)
    p_c4k = p_k4c.transpose()
    p_k4c = p_k4c / p_k4c.sum(axis=0)
    p_k4c[np.isnan(p_k4c)] = 1
    p_c4k = p_c4k / p_c4k.sum(axis=0)
    p_c4k[np.isnan(p_c4k)] = 1
    np.seterr(**old_settings)

    # p_k4c = np.exp(cluster_colors_diffs)
    # p_c4k = p_k4c.transpose()
    # p_k4c = p_k4c / p_k4c.sum(axis=0)
    # p_c4k = p_c4k / p_c4k.sum(axis=0)

    return p_k4c, p_c4k


def calculateMeanVisual(visual, idx):
    new_visual = np.zeros_like(visual)
    for ii in range(new_visual.shape[0]):
        for jj in range(new_visual.shape[1]):
            new_visual[ii, jj] = visual[idx[ii, jj]].mean()
    return new_visual


def truncatedMinimalNeighbors(visual, t):
    """
    TODO try other neighborhood definitions
    """

    padded_visual = np.pad(visual, pad_width=(1, 1), mode='constant', constant_values=(np.inf, np.inf))

    # for 8 neighbors of each pixel, calculate the abs(diff)
    diffs = np.zeros(visual.shape + (8,))
    diffs[:, :, 0] = abs(padded_visual[: -2, 1: -1] - visual)  # up
    diffs[:, :, 1] = abs(padded_visual[2:, 1: -1] - visual)  # down
    diffs[:, :, 2] = abs(padded_visual[1: -1, : -2] - visual)  # left
    diffs[:, :, 3] = abs(padded_visual[1: -1, 2:] - visual)  # right
    diffs[:, :, 4] = abs(padded_visual[: -2, : -2] - visual)  # up_left
    diffs[:, :, 5] = abs(padded_visual[2:, 2:] - visual)  # down_right
    diffs[:, :, 6] = abs(padded_visual[: -2, 2:] - visual)  # up_right
    diffs[:, :, 7] = abs(padded_visual[2:, : -2] - visual)  # down_left

    min_idx = diffs.argmin(axis=2)
    X, Y = np.meshgrid(np.arange(visual.shape[0]), np.arange(visual.shape[1]), indexing='ij')
    mins = diffs[(X, Y, min_idx)]
    neighbor_idx = np.empty_like(mins, dtype=tuple)
    for ii in range(mins.shape[0]):
        for jj in range(mins.shape[1]):
            if mins[ii, jj] > t:
                neighbor_idx[ii, jj] = ([ii], [jj])
            else:
                neighbor_idx[ii, jj] = {0: ([ii, ii - 1], [jj, jj]),
                                        1: ([ii, ii + 1], [jj, jj]),
                                        2: ([ii, ii], [jj, jj - 1]),
                                        3: ([ii, ii], [jj, jj + 1]),
                                        4: ([ii, ii - 1], [jj, jj - 1]),
                                        5: ([ii, ii + 1], [jj, jj + 1]),
                                        6: ([ii, ii - 1], [jj, jj + 1]),
                                        7: ([ii, ii + 1], [jj, jj - 1])}[min_idx[ii, jj]]
                # neighbor_idx[ii, jj] = {0: ([ii - 1], [jj]),
                #                         1: ([ii + 1], [jj]),
                #                         2: ([ii], [jj - 1]),
                #                         3: ([ii], [jj + 1]),
                #                         4: ([ii - 1], [jj - 1]),
                #                         5: ([ii + 1], [jj + 1]),
                #                         6: ([ii - 1], [jj + 1]),
                #                         7: ([ii + 1], [jj - 1])}[min_idx[ii, jj]]

    return neighbor_idx


def canStop(old_visual, old_centers, new_visual, new_centers):
    print("new_centers: {}".format(new_centers))
    return np.all(old_visual == new_visual)  # and np.all(old_centers == new_centers)


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
                neighbor_idx[ii, jj] = (valid_ii_idx, valid_jj_idx)
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
                neighbor_idx[ii, jj] = (valid_ii_idx, valid_jj_idx)
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
                neighbor_idx[ii, jj] = (valid_ii_idx, valid_jj_idx)
    else:
        return None

    return neighbor_idx


def initializeClusterByColor(obs_visual_h, predefined_colors_h):

    predefined_colors_bins = {k: np.digitize(v, bin_edges) for (k, v) in predefined_colors_h.items()}

    # initialize bin(pixel_idx, sizes)
    cluster_pixel_idx = []
    bin_sizes = np.zeros(n_bins, dtype=int)
    bin_labels = np.digitize(obs_visual_h, bin_edges)
    for bin_id in range(n_bins):
        cluster_pixel_idx.append(np.array(np.where(bin_labels == bin_id + 1)))
        bin_sizes[bin_id] = cluster_pixel_idx[bin_id].shape[1]

    # update bin(pixel_idx, sizes): merge small bins into large bins
    for bin_id in range(n_bins):
        if bin_sizes[bin_id] > bin_size_limit \
                or (bin_id in predefined_colors_bins.values()):
            continue

        delta = 1
        while True:
            bin_id_tmp = (bin_id - delta) % n_bins
            if bin_sizes[bin_id_tmp] > bin_size_limit \
                    or (bin_id in predefined_colors_bins.values()):
                bin_sizes[bin_id_tmp] += bin_sizes[bin_id]
                bin_sizes[bin_id] = 0
                cluster_pixel_idx[bin_id_tmp] = np.concatenate(
                    (cluster_pixel_idx[bin_id], cluster_pixel_idx[bin_id_tmp]), axis=1)
                break

            bin_id_tmp = (bin_id + delta) % n_bins
            if bin_sizes[bin_id_tmp] > bin_size_limit \
                    or (bin_id in predefined_colors_bins.values()):
                bin_sizes[bin_id_tmp] += bin_sizes[bin_id]
                bin_sizes[bin_id] = 0
                cluster_pixel_idx[bin_id_tmp] = np.concatenate(
                    (cluster_pixel_idx[bin_id], cluster_pixel_idx[bin_id_tmp]), axis=1)
                break

            delta += 1

    cluster_pixel_idx = [cluster_pixel_idx[ii] for ii in np.where(bin_sizes != 0)[0]]
    cluster_centers = np.zeros(len(cluster_pixel_idx), dtype=float)
    for cluster_id in range(len(cluster_centers)):
        cluster_centers[cluster_id] = obs_visual_h[tuple(cluster_pixel_idx[cluster_id])].mean(axis=0)

    return cluster_pixel_idx, cluster_centers


four_neighbor_idx = orthogonalNeighbors(4)
eight_neighbor_idx = orthogonalNeighbors(8)
twel_neighbor_idx = orthogonalNeighbors(12)
