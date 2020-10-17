import numpy as np
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


def d_a(l1, b1, l2, b2):
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2)
    b2 = np.deg2rad(b2)

    d_a = np.arccos(np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2))

    return np.rad2deg(d_a)


def two_d(ra, dec):
    two_d = np.zeros((len(ra), len(ra)))

    for i in tqdm(range(0, len(ra)), ascii=True, desc='two_d'):
        two_d[i, :] = np.array(d_a(ra[i], dec[i], ra, dec))

    return two_d


def fof(n_min, n_max, lk, distance, sample_size):
    rem_index = np.arange(sample_size).tolist()
    j, i = np.meshgrid(np.arange(len(rem_index)), np.arange(len(rem_index)))

    ##############################Friends_of_Friends###############################
    groups = (np.zeros((len(rem_index), 1)) - 1).tolist()

    start = time.time()
    while len(rem_index) > 0:
        #############index[0]&its_neighbors##############
        index = rem_index[0]
        groups[index] = [index]
        neighbors = np.where((distance[:, index] > 0) &
                             (distance[:, index] < lk))[0].tolist()
        groups[index] = groups[index] + neighbors
        if __name__ == '__main__':
            if len(neighbors) > 0:
                print("Index[%i]'s neighbors: %s" % (index, neighbors))
            else:
                print("Index[%i] has no neighbors" % (index))
        #########Remove_ind[0]&its_neighbors###########
        rem_index = list(set(rem_index) - set(groups[index]))

        #########neighbors'_neighbors###########
        while len(neighbors) > 0:
            index_n = neighbors[0]
            neighbors_neighbors = np.where((distance[:, index_n] > 0) &
                                           (distance[:, index_n] < lk))[0].tolist()
            neighbors_neighbors = list(set(neighbors_neighbors) & set(rem_index))
            neighbors_neighbors = [x for x in neighbors_neighbors if x in rem_index]
            if len(neighbors_neighbors) == 0:
                if __name__ == '__main__':
                    print("Neighbor[%i] has no neighbor." % (neighbors[0]))
            else:
                if __name__ == '__main__':
                    print("Neighbor[%i] has neighbors: %s" %
                          (neighbors[0], sorted(neighbors_neighbors)))
                groups[index] = groups[index] + neighbors_neighbors
                rem_index = list(set(rem_index) - set(neighbors_neighbors))
                rem_index = [x for x in rem_index if x not in neighbors_neighbors]
                neighbors = neighbors + neighbors_neighbors

            neighbors.remove(neighbors[0])

    if __name__ == '__main__':
        print("groups[%i]:%s" % (index, sorted(groups[index])))

    print("FoF:%fs" % (time.time() - start))

    groups = sorted([row for row in groups if ((len(row) >= n_min) & (len(row) <= n_max))], key=len, reverse=True)
    print("Groups: %i" % (len(groups)))

    members = []
    for i in range(len(groups)):
        members = members + groups[i]

    print("Members: %i" % (len(members)))

    return groups


def ezfof(coord, sep=5):
    """
    mark the group ids

    Parameters
    ----------
    coord:
        SkyCoord
    sep:
        separation in deg

    Returns
    -------
    gid:
        group id

    """
    npts = len(coord)
    cid = np.arange(npts, dtype=int)
    gids = - np.ones_like(cid)
    cgid = 0  # current group id
    while np.sum(gids < 0) > 0:
        # set seed for this group
        ind_rest = np.where(gids < 0)[0]
        if np.sum(gids < 0) == 0:
            break
        gids[ind_rest[0]] = cgid

        while True:
            ind_thisGroup = np.where(gids == cgid)[0]
            ind_thisGroup_extend = np.unique(
                np.hstack([np.where(coord[_].separation(coord).deg < sep)[0] for _ in ind_thisGroup]))

            if ind_thisGroup_extend.shape[0] == ind_thisGroup.shape[0]:
                break
            else:
                gids[ind_thisGroup_extend] = cgid
                continue
        cgid += 1
    return gids


if __name__ == "__main__":
    import astropy.table as ast
    import numpy as np

    path = r'/Users/yang/Desktop/'
    data = ast.Table.read(path + 'test.fits')[:1000]

    twod = two_d(data['ra'], data['dec'])

    groups = fof(n_min=1, n_max=10000000, lk=1, distance=twod, sample_size=len(data))
