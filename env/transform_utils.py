""" Helper functions to compute transformation of position and rotation. """
""" Many functions from https://github.com/StanfordVL/robosuite/blob/master/robosuite/utils/transform_utils.py """

import math

import numpy as np
from pyquaternion import Quaternion


_PI = np.pi
_EPS = np.finfo(float).eps * 4.0


# code from stanford robosuite
def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    Examples:

        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quat_conjugate(quaternion):
    """Return conjugate of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )


def quat_inverse(quaternion):
    """Return inverse of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    d = min(d, 1.0)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quat(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )


def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose: a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float32, copy=True)[[3, 0, 1, 2]]
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )


def pose_inv(pose):
    """
    Computes the inverse of a homogenous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose: numpy array of shape (4,4) for the pose to inverse

    Returns:
        numpy array of shape (4,4) for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A: numpy array of shape (4,4) corresponding to the pose of C in frame A
        pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B

    Returns:
        numpy array of shape (4,4) corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)


def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat: a 4x4 homogeneous matrix

    Returns:
        (pos, orn) tuple where pos is vec3 float in cartesian,
            orn is vec4 float quaternion
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if precise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
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
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q[[1, 2, 3, 0]]


def euler2mat(euler):  # assume xyz
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    Examples:

        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True
        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def make_pose(translation, rotation):
    """
    Makes a homogenous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation: a 3-dim iterable
        rotation: a 3x3 matrix

    Returns:
        pose: a 4x4 homogenous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def norm(x):
    """ Returns the normalized vector """
    return x / np.linalg.norm(x)


def lookat_to_quat(forward, up):
    """
    Converts forward and up vectors to a quaternion

    Args:
        forward: a 3-dim forward vector
        up: a 3-dim up vector

    Returns:
        a 4-dim quaternion
    """
    vector = norm(forward)
    vector2 = norm(np.cross(norm(up), vector))
    vector3 = np.cross(vector, vector2)
    m00 = vector2[0]
    m01 = vector2[1]
    m02 = vector2[2]
    m10 = vector3[0]
    m11 = vector3[1]
    m12 = vector3[2]
    m20 = vector[0]
    m21 = vector[1]
    m22 = vector[2]

    num8 = (m00 + m11) + m22
    quaternion = np.zeros(4)
    if num8 > 0:
        num = np.sqrt(num8 + 1)
        quaternion[3] = num * 0.5
        num = 0.5 / num
        quaternion[0] = (m12 - m21) * num
        quaternion[1] = (m20 - m02) * num
        quaternion[2] = (m01 - m10) * num
        return quaternion

    if (m00 >= m11) and (m00 >= m22):
        num7 = np.sqrt(((1 + m00) - m11) - m22)
        num4 = 0.5 / num7
        quaternion[0] = 0.5 * num7
        quaternion[1] = (m01 + m10) * num4
        quaternion[2] = (m02 + m20) * num4
        quaternion[3] = (m12 - m21) * num4
        return quaternion

    if m11 > m22:
        num6 = np.sqrt(((1 + m11) - m00) - m22)
        num3 = 0.5 / num6
        quaternion[0] = (m10 + m01) * num3
        quaternion[1] = 0.5 * num6
        quaternion[2] = (m21 + m12) * num3
        quaternion[3] = (m20 - m02) * num3
        return quaternion

    num5 = np.sqrt(((1 + m22) - m00) - m11)
    num2 = 0.5 / num5
    quaternion[0] = (m20 + m02) * num2
    quaternion[1] = (m21 + m12) * num2
    quaternion[2] = 0.5 * num5
    quaternion[3] = (m01 - m10) * num2
    return quaternion


# https://www.gamedev.net/forums/topic/56471-extracting-direction-vectors-from-quaternion/
def forward_vector_from_quat(quat):
    """ Returns a forward vector of a quaternion """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qy + qw * qz)
    y = 1 - 2 * (qx * qx + qz * qz)
    z = 2 * (qy * qz - qw * qx)
    return np.array([x, y, z])


def up_vector_from_quat(quat):
    """ Returns a up vector of a quaternion """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qz - qw * qy)
    y = 2 * (qy * qz + qw * qx)
    z = 1 - 2 * (qx * qx + qy * qy)
    return np.array([x, y, z])


def right_vector_from_quat(quat):
    """ Returns a right vector of a quaternion """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 1 - 2 * (qy * qy + qz * qz)
    y = 2 * (qx * qy - qw * qz)
    z = 2 * (qx * qz + qw * qy)
    return np.array([x, y, z])


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.
    Examples:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z


def euler_to_quat(rotation, quat=None):
    """ Returns a quaternion of a euler rotation """
    q1 = Quaternion(axis=[1, 0, 0], degrees=rotation[0])
    q2 = Quaternion(axis=[0, 1, 0], degrees=rotation[1])
    q3 = Quaternion(axis=[0, 0, 1], degrees=rotation[2])
    q = q3 * q2 * q1
    # q = Quaternion(
    #     convert_quat(mat2quat(euler2mat(np.array(rotation) / 180.0 * np.pi)), to="wxyz")
    # )
    if quat is None:
        final_quat = list(q)
    else:
        final_quat = list(Quaternion(quat) * q)
    return final_quat


def rel_pose(qpos1, qpos2):
    """ Returns relative pose of @qpos2 w.r.t @qpos1 """
    rel_quat = Quaternion(qpos1[3:]).inverse * Quaternion(qpos2[3:])
    rel_pos = qpos2[:3] - qpos1[:3]
    rel_pos = Quaternion(qpos1[3:]).inverse.rotate(rel_pos)
    return np.concatenate([rel_pos, list(rel_quat)])


def transform_to_target_quat(qpos_base, qpos, target_quat):
    """
    Returns new position and rotation of @qpos when rotate @qpos_base to @target_quat

    Args:
        qpos_base: qpos (position and rotation) of the base point
        qpos: qpos (position and rotation) to be rotated around the base point
        target_quat: the target rotation of the base point

    Returns:
        new_pos: position of @qpos when rotated around the base point
        new_quat: rotation of @qpos when rotated around the base point
    """
    cur_pos = qpos_base[:3]
    cur_rot = Quaternion(qpos_base[3:])

    pos = qpos[:3]
    rot = Quaternion(qpos[3:])

    rel_rot = Quaternion(target_quat) * cur_rot.inverse

    new_pos = rel_rot.rotate(pos - cur_pos) + cur_pos
    new_rot = rel_rot * rot
    return new_pos, list(new_rot)


def l2_dist(a, b):
    """ Returns l2 distance between vectors @a and @b """
    return np.linalg.norm(a - b)


def angle_between2D(cur, tgt):
    """ Returns the relative angle in radians between 2D vectors 'cur' and 'tgt'::

            >>> angle_between2D((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between2D((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between2D((1, 0, 0), (-1, -1, 0))
            3.9269908169872414
    """
    cur_u = unit_vector(cur)
    tgt_u = unit_vector(tgt)
    return math.atan2(cur_u[0], cur_u[1]) - math.atan2(tgt_u[0], tgt_u[1])


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def cos_dist(a, b):
    """ Returns cos distance between vectors @a and @b """
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


def up_vector_cos_dist(quat1, quat2):
    """ Returns cos distance between up vectors of quaternions @quat1 and @quat2 """
    up1 = up_vector_from_quat(quat1)
    up2 = up_vector_from_quat(quat2)
    return cos_dist(up1, up2)


def rotate_vector(v, rotation_axis, angle):
    """ Returns a vector rotating @v @angle degree along @rotation_axis  """
    v = np.asarray(v)
    k = unit_vector(rotation_axis)
    angle = angle / 180 * _PI
    new_v = np.cos(angle) * v + np.sin(angle) * np.cross(k, v)
    return new_v


def rotate_vector_cos_dist(v, rotation_axis, cos, direction):
    """ Returns a vector rotating @v an angle of @cos along @rotation_axis """
    assert direction in [-1, 1]
    v = np.asarray(v)
    k = unit_vector(rotation_axis)
    new_v = cos * v + direction * np.sqrt(1 - cos ** 2) * np.cross(k, v)
    return new_v
