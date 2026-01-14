import cv2 as cv
import numpy as np
import matrix as mat



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


def new(name, size, mouse_cb=None) -> np.ndarray:
    if len(size) == 2:
        size = (*size, 3)
    cv.namedWindow(name)
    if mouse_cb is not None:
        cv.setMouseCallback(name, mouse_cb)
    canvas = np.zeros(size, dtype='uint8')
    _windows[name] = canvas
    return canvas


def drawPoints(canvas, pts):
    for pt1, pt2 in zip(pts[:-1], pts[1:]):
        canvas = cv.circle(canvas, np.int32(pt1), 14, (0, 255, 0), 3)
        canvas = cv.line(canvas, np.int32(pt1), np.int32(pt2), (0, 0, 255), 10)
    return canvas


def drawChain(canvas, base, segments, target):
    # base
    canvas = cv.rectangle(canvas, np.int32(base - 30), np.int32(base + 30), (0, 255, 255), 3)
    # segments
    pts = mat.forward(segments)
    pts = [pt + base for pt in pts]
    canvas = drawPoints(canvas, pts)
    # target
    canvas = cv.circle(canvas, target, 15, (255,255,255), 3)
    return canvas


def drawBoxes(canvas, base, boxes):
    for pt1, pt2 in boxes:
        canvas = cv.rectangle(canvas, np.int32(pt1 + base), np.int32(pt2 + base), (255,128,128), 3)
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
        if np.linalg.norm(b - e, np.inf) < canvas.shape[0] / 2:
            canvas = cv.line(canvas, b, e, (255,0,0), 1)

    return canvas
