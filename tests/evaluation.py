import math
import cv2
import numpy as np


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def qvec2rotmat(qvec):
    """ Convert from quaternions to rotation matrix. """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotation_error_from_trace(R_gt, R, eps=1e-15):
    """
    Returns the rotation error in degrees using the trace of R_gt^T R.

    Args:
        R_gt: (..., 3, 3) array-like ground-truth rotation(s).
        R:    (..., 3, 3) array-like estimated rotation(s).
        eps:  small constant for numerical stability.

    Returns:
        err_R_deg: (...) ndarray of rotation errors in degrees.
    """
    R_gt = np.asarray(R_gt, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Relative rotation: R_rel = R_gt^T R
    # angle = arccos( (trace(R_rel)-1)/2 )
    R_rel = np.matmul(np.swapaxes(R_gt, -1, -2), R)
    tr = np.trace(R_rel, axis1=-2, axis2=-1)

    # Clamp to valid range for acos
    c = (tr - 1.0) * 0.5
    c = np.clip(c, -1.0 + eps, 1.0 - eps)

    err_rad = np.arccos(c)
    err_deg = np.degrees(err_rad)
    return err_deg

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    if t is None or R is None:
        return 180.0, 180.0
    t = np.asarray(t, dtype=np.float64).flatten()
    t_gt = np.asarray(t_gt, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)

    eps = 1e-15

    if not np.all(np.isfinite(R)) or R.shape != (3, 3):
        err_q = 180.0
    else:
        err_q = float(rotation_error_from_trace(R_gt, R))
        if np.isnan(err_q):
            err_q = 180.0

    norm_t = np.linalg.norm(t)
    norm_t_gt = np.linalg.norm(t_gt)
    if norm_t < eps or norm_t_gt < eps or not np.all(np.isfinite(t)):
        err_t = 180.0
    else:
        cos_t = np.clip(np.dot(t / norm_t, t_gt / norm_t_gt), -1.0, 1.0)
        err_t = float(np.degrees(np.arccos(cos_t)))
        if np.isnan(err_t):
            err_t = 180.0

    return err_q, err_t


def homography_pose_error(H, scale, pose_gt, K1, K2):
    if H is None or not np.all(np.isfinite(H)) or np.linalg.norm(H) < 1e-12:
        return 180.0, 180.0, np.inf
    R_gt = pose_gt[:, :3]
    t_gt = pose_gt[:, 3]
    try:
        num, Rs, ts, _ = cv2.decomposeHomographyMat(np.asarray(H, dtype=np.float64), np.asarray(K2, dtype=np.float64))
    except cv2.error:
        return 180.0, 180.0, np.inf
    if num == 0 or not Rs:
        return 180.0, 180.0, np.inf
    best_max = 180.0
    best_r, best_t = 180.0, 180.0
    for R_cand, t_cand in zip(Rs, ts):
        er, et = evaluate_R_t(R_gt, t_gt, R_cand, t_cand)
        if max(er, et) < best_max:
            best_max = max(er, et)
            best_r, best_t = er, et
    return best_r, best_t, best_t


def pose_auc(errors, thresholds=[5, 10, 20]):
    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) == 0:
        return [0.0 for _ in thresholds]
    errors = np.where(np.isnan(errors) | np.isinf(errors), 1e9, errors)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(float(trapz_fn(r, x=e) / t))
    return aucs
