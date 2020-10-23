import cv2
from networks import *
from sklearn.neighbors import KernelDensity


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]



def estimate_partial_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process

    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    prev_matched_kp, cur_matched_kp = matched_keypoints

    # transform = cv2.estimateRigidTransform(np.array(prev_matched_kp),
    #                                        np.array(cur_matched_kp),
    #                                        False)
    transform = cv2.estimateAffinePartial2D(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp))[0]

    if transform is not None:
        # translation x
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]


def check_dy_dx_da(dy, dx, da, d_max=20.0, d_min=-20.0):

    if dy > d_max: dy = d_max
    if dy < d_min: dy = d_min
    if dx > d_max: dx = d_max
    if dx < d_min: dx = d_min
    if da > d_max: da = d_max
    if da < d_min: da = d_min

    return dy, dx, da



def removeOutliers(prev_pts, curr_pts):

    d = np.sum((prev_pts - curr_pts)**2, axis=-1)**0.5

    d_ = np.array(d).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(d_)
    density = np.exp(kde.score_samples(d_))

    prev_pts = prev_pts[np.where((density >= 0.1))]
    curr_pts = curr_pts[np.where((density >= 0.1))]

    return prev_pts, curr_pts

