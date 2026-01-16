import cv2 as cv
import numpy as np
import matrix as mat
from enum import Enum



class Colors(Enum):
    box             = 48,48,48,
    box_outline     = 112,112,112,
    joint           = 192,32,64,
    joint_outline   = 255,64,128,
    segment         = 32,128,160,
    segment_outline = 64,255,255,
    target          = 255,255,255,
    edge            = 255,0,0,


_windows: dict[str, np.ndarray] = {}


def nothing(_): pass


def close():
    cv.destroyAllWindows()


def showKey(ms: int = 10) -> bool:
    for name, img in _windows.items():
        cv.imshow(name, img)
    return cv.waitKey(ms)


def show(ms: int = 10) -> bool:
    return showKey(ms) != 27


def new(name, size, mouse_cb=None, sliders: list[str] = []) -> np.ndarray:
    if len(size) == 2:
        size = (*size, 3)
    cv.namedWindow(name)
    if mouse_cb is not None:
        cv.setMouseCallback(name, mouse_cb)
    for slider in sliders:
        cv.createTrackbar(slider, name, 0, 100, nothing)
    canvas = np.zeros(size, dtype='uint8')
    _windows[name] = canvas
    return canvas


def drawGradient(canvas, strength: float):
    xv = np.linspace(0, 255, canvas.shape[1])
    yv = np.linspace(0, 255, canvas.shape[0])
    xvv, yvv = np.meshgrid(xv, yv)
    canvas[:,:,2] = np.int8(xvv * strength)
    canvas[:,:,1] = np.int8(yvv * strength)
    canvas[:,:,0] &= 0
    return canvas


def drawPoints(canvas, pts):
    for pt1, pt2 in zip(pts[:-1], pts[1:]):
        canvas = cv.circle(canvas, np.int32(pt1), 15, Colors.joint_outline.value, 9)
        canvas = cv.circle(canvas, np.int32(pt1), 15, Colors.joint.value, 5)
        canvas = cv.line(canvas, np.int32(pt1), np.int32(pt2), Colors.segment_outline.value, 14)
        canvas = cv.line(canvas, np.int32(pt1), np.int32(pt2), Colors.segment.value, 10)
    return canvas


def drawChain(canvas, base, segments, target):
    # segments
    pts = mat.forward(segments)
    pts = [pt + base for pt in pts]
    canvas = drawPoints(canvas, pts)
    # target
    canvas = cv.circle(canvas, target, 15, Colors.target.value, 3)
    return canvas


def drawBoxes(canvas, base, boxes):
    for pt1, pt2 in boxes:
        canvas = cv.rectangle(canvas, np.int32(pt1 + base), np.int32(pt2 + base), Colors.box.value, -1)
        canvas = cv.rectangle(canvas, np.int32(pt1 + base), np.int32(pt2 + base), Colors.box_outline.value, 3)
    return canvas


def drawShortestLine(canvas, b, e, c=Colors.edge.value):
    if np.linalg.norm(b - e, np.inf) < canvas.shape[0] / 2:
            canvas = cv.line(canvas, b, e, c, 1)
    
    elif abs(b[0] - e[0]) > canvas.shape[0] / 2 and abs(b[1] - e[1]) < canvas.shape[1] / 2:
        if b[0] > e[0]: b, e = e, b
        off_x = np.array([canvas.shape[0], 0])
        canvas = cv.line(canvas, b, e - off_x, c, 1)
        canvas = cv.line(canvas, b + off_x, e, c, 1)

    elif abs(b[1] - e[1]) > canvas.shape[1] / 2 and abs(b[0] - e[0]) < canvas.shape[0] / 2:
        if b[1] > e[1]: b, e = e, b
        off_y = np.array([0, canvas.shape[1]])
        canvas = cv.line(canvas, b, e - off_y, c, 1)
        canvas = cv.line(canvas, b + off_y, e, c, 1)

    else:
        if b[0] > e[0]: b, e = e, b
        off_x = np.array([canvas.shape[0], 0])
        off_y = np.array([0, canvas.shape[1]])
        if b[1] > e[1]:
            canvas = cv.line(canvas, b, e - off_x + off_y, c, 1)
            canvas = cv.line(canvas, b + off_x - off_y, e, c, 1)
        else:
            canvas = cv.line(canvas, b, e - off_x - off_y, c, 1)
            canvas = cv.line(canvas, b + off_x + off_y, e, c, 1)
    return canvas


def drawGraph(canvas, edges):
    size = np.array([canvas.shape[1], canvas.shape[0]])
    scale = size / 2 / np.pi
    center = np.int32(size // 2)
    
    def pmap(p):
        tp = np.int32(p * scale) + center
        return np.array([tp[0] % size[0], tp[1] % size[1]])
    
    for begin, end in edges:
        b = pmap(begin)
        e = pmap(end)
        canvas = drawShortestLine(canvas, b, e, c=Colors.edge.value)
    return canvas
