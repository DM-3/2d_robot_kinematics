import numpy as np



def rot(angle):
    return np.array([[np.cos(angle),  np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [0,              0,             1]])


def trans(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def forward(segments):
    pts = [np.zeros(2)]
    mat = np.eye(3)
    for angle, length in segments:
        mat = mat @ rot(angle) @ trans(length, 0)
        pts.append(mat[:2,2])
    return pts


# Forward And Backward Reaching Inverse Kinematics
def FABRIK_step(target, base, segments):

    pts = forward(segments)
    pts = [pt + base for pt in pts]

    # backward pass
    pts[-1] = target
    for i in range(len(segments) - 1, 0, -1):
        length = segments[i][1]
        vec = pts[i] - pts[i + 1]
        vec /= np.linalg.norm(vec)
        pts[i] = pts[i + 1] + vec * length

    # forward pass
    for i in range(len(segments)):
        length = segments[i][1]
        vec = pts[i + 1] - pts[i]
        vec /= np.linalg.norm(vec)
        pts[i + 1] = pts[i] + vec * length

    # recompute angles
    prev = 0
    for i in range(len(segments)):
        vec = pts[i + 1] - pts[i]
        angle = np.arctan2(-vec[1], vec[0])
        segments[i] = (angle - prev, segments[i][1])
        prev = angle

    return segments
