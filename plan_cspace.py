import numpy as np
import cv2 as cv
import time
import matrix as mat
import ui
from planners import *
import argparse



def ptCollidesWithBox(x, y, box):
    pt1, pt2 = box
    return x >= pt1[0] and x <= pt2[0] and y >= pt1[1] and y <= pt2[1]


def ptCollidesWithBoxes(x, y, boxes):
    for box in boxes:
        if ptCollidesWithBox(x, y, box):
            return True
    return False


def ptsCollideWithBoxes(pts, boxes):
    if len(pts.shape) != 3:
        raise NotImplementedError
    
    collisions = np.zeros((pts.shape[:2]), dtype='bool')
    for pt1, pt2 in boxes:
        collisions |= np.logical_and(np.logical_and(pts[:,:,0] >= pt1[0], pts[:,:,0] <= pt2[0]),
                                     np.logical_and(pts[:,:,1] >= pt1[1], pts[:,:,1] <= pt2[1]))        
    return collisions


def configCollidesWithBoxes(q, segments, boxes):
    s = [(q[i], seg[1]) for i, seg in enumerate(segments)]
    pts = mat.forward(s)
    p0 = np.array([[pts[0]]])
    p1 = np.array([[pts[1]]])
    p2 = np.array([[pts[2]]])
    c = linesCollideWithBoxes([p0, p1, p2], boxes)
    return c[0,0]


def linesCollideWithBoxes(lines, boxes):
    if len(lines[0].shape) != 3:
        raise NotImplementedError
    
    collisions = np.zeros((lines[0].shape[:2]), dtype='bool')

    for begin_pts, end_pts in zip(lines[:-1], lines[1:]):

        m = (end_pts[:,:,1] - begin_pts[:,:,1]) / (end_pts[:,:,0] - begin_pts[:,:,0])
        min_x = np.minimum(begin_pts[:,:,0], end_pts[:,:,0])
        max_x = np.maximum(begin_pts[:,:,0], end_pts[:,:,0])
        min_y = np.minimum(begin_pts[:,:,1], end_pts[:,:,1])
        max_y = np.maximum(begin_pts[:,:,1], end_pts[:,:,1])
        
        for pt1, pt2 in boxes:
            x1 = begin_pts[:,:,0] + (pt1[1] - begin_pts[:,:,1]) / m     # intersection with top line of box
            collisions |= np.logical_and(np.logical_and(x1 >= pt1[0], x1 <= pt2[0]),
                                         np.logical_and(x1 >= min_x, x1 <= max_x))
            x2 = begin_pts[:,:,0] + (pt2[1] - begin_pts[:,:,1]) / m     # intersection with bottom line of box
            collisions |= np.logical_and(np.logical_and(x2 >= pt1[0], x2 <= pt2[0]),
                                         np.logical_and(x2 >= min_x, x2 <= max_x))
            y1 = begin_pts[:,:,1] + (pt1[0] - begin_pts[:,:,0]) * m     # intersection with top line of box
            collisions |= np.logical_and(np.logical_and(y1 >= pt1[1], y1 <= pt2[1]),
                                         np.logical_and(y1 >= min_y, y1 <= max_y))
            y2 = begin_pts[:,:,1] + (pt2[0] - begin_pts[:,:,0]) * m     # intersection with bottom line of box
            collisions |= np.logical_and(np.logical_and(y2 >= pt1[1], y2 <= pt2[1]),
                                         np.logical_and(y2 >= min_y, y2 <= max_y))

    return collisions


def dkin_all_angles(segments, resolution):
    if len(segments) > 2:
        raise NotImplementedError("currently can't handle more than 2 segments")

    pos_v = np.zeros([resolution, resolution, 2])
    
    angle_l = [np.linspace(-np.pi, np.pi, resolution) for _ in segments]
    sin_l = np.sin(angle_l)
    cos_l = np.cos(angle_l)
    sin_v = np.meshgrid(*sin_l)
    cos_v = np.meshgrid(*cos_l)

    for (_, length), sin, cos in zip(segments[::-1], sin_v[::-1], cos_v[::-1]):
        pos_v[:,:,0] += length                          # translate by segment length along x
        nx =  cos * pos_v[:,:,0] + sin * pos_v[:,:,1]   # rotate by all angles
        ny = -sin * pos_v[:,:,0] + cos * pos_v[:,:,1]
        pos_v[:,:,0] = nx
        pos_v[:,:,1] = ny

    return pos_v


wsize = (800, 800)
base = np.array(wsize) / 2
segments = [
    (0, 90),
    (0, 250)
]

boxes = [
    (np.array([-100, -300]), np.array([200, -80])),
    (np.array([-300, 200]), np.array([-100, 300]))
]

target = np.int32(base) + np.array([-200, 0])
grab_relative = np.array([0, 0])
mouse_grab = 0
def on_mouse(event,x,y,flags,param):
    global target, boxes, grab_relative, mouse_grab
    if event == cv.EVENT_LBUTTONDOWN:
        if np.linalg.norm(np.array([x,y]) - target) < 17:
            mouse_grab = 1
        for i, box in enumerate(boxes):
            if ptCollidesWithBox(x - base[0], y - base[1], box):
                mouse_grab = i + 2
                grab_relative = box[0] - np.array([x, y]) + base
    if event == cv.EVENT_MOUSEMOVE and mouse_grab > 0:
        if mouse_grab == 1:
            target = np.array([x, y])
        else:
            i = mouse_grab - 2
            bsize = boxes[i][1] - boxes[i][0]
            pt1 = np.array([x, y]) - base + grab_relative
            boxes[i] = (pt1, pt1 + bsize)
    if event == cv.EVENT_LBUTTONUP:
        mouse_grab = 0

canvas = ui.new("scene", wsize, on_mouse, ["cartesian xy"])
dkin_plot = np.zeros((360, 360, 3), dtype='uint8')

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--planner", default="prm", help="""options:
    rrt (Rapid Random Tree),
    prm (Probabilistic Roadmap)
    """)
args = parser.parse_args()

planner = None
path = []

while True:
    # execute planned path
    if len(path) > 0:
        q = path.pop()
        segments = [(q[i], seg[1]) for i, seg in enumerate(segments)]

    # draw scene
    canvas = ui.drawGradient(canvas, cv.getTrackbarPos("cartesian xy", "scene") * .01)
    canvas = ui.drawBoxes(canvas, base, boxes)
    canvas = ui.drawChain(canvas, base, segments, target)

    # draw C-space plot
    dkin_pos_v = dkin_all_angles(segments, 360)   # chain end positions for all possible angle combinations under the given resolution
    dkin_pos1_v = dkin_all_angles(segments[:-1], 360)
    dkin_pos0_v = np.zeros_like(dkin_pos_v)

    # plot end positions
    dkin_plot &= 0
    dkin_plot[:,:,2] = np.int8((dkin_pos_v[:,:,0] + base[0]) / wsize[0] * 255 * cv.getTrackbarPos("cartesian xy", "scene") * .01)
    dkin_plot[:,:,1] = np.int8((dkin_pos_v[:,:,1] + base[1]) / wsize[1] * 255 * cv.getTrackbarPos("cartesian xy", "scene") * .01)

    # plot box colliders
    bmask = np.zeros_like(dkin_plot, dtype='bool')
    bmask[:,:,0] = bmask[:,:,1] = bmask[:,:,2] = linesCollideWithBoxes([dkin_pos0_v, dkin_pos1_v, dkin_pos_v], boxes)
    dkin_plot *= np.uint8(~bmask)
    dkin_plot += np.ones_like(dkin_plot) * bmask * np.where(
        cv.erode(np.uint8(bmask), np.ones((3,3), np.uint8)) > 0, 
        np.uint8(ui.Colors.box.value[0]), 
        np.uint8(ui.Colors.box_outline.value[0]))

    # plot solution curve
    smask = np.zeros_like(dkin_plot, dtype='bool')
    smask[:,:,0] = np.abs(dkin_pos_v[:,:,0] - (target[0] - base[0])) + \
                  np.abs(dkin_pos_v[:,:,1] - (target[1] - base[1])) < 8.0
    smask[:,:,1] = smask[:,:,2] = smask[:,:,0]
    dkin_plot = np.where(smask, np.ones_like(dkin_plot) * ui.Colors.target.value[0], dkin_plot)
    
    # plot current configuration
    cur_ang = (np.int32(np.degrees(segments[0][0]) + 540) % 360, 
               np.int32(np.degrees(segments[1][0]) + 540) % 360)
    dkin_plot = cv.circle(dkin_plot, (cur_ang), 3, ui.Colors.joint_outline.value, 2)

    if isinstance(planner, Planner):
        for _ in range(10):
            planner.iterate()
        ui.drawGraph(dkin_plot, planner.edges)
    
    res_dkin_plot = cv.resize(dkin_plot, (720, 720))
    cv.imshow("C space", res_dkin_plot)

    key = ui.showKey(1)
    if key == 27:
        break
    
    # plan path
    if key == ord('p'):
        pstart = np.array([segments[0][0], segments[1][0]])
        invkin_options = cv.findNonZero(np.where(smask[:,:,0] & ~bmask[:,:,0], np.uint8(1), np.uint8(0)))
        if invkin_options is None:
            continue
        ptarget = (invkin_options[0][0] - 180.) / 180. * np.pi
        print("planning request from:", pstart, " to:",ptarget)
        pshape = np.array([2 * np.pi, 2 * np.pi])
        if args.planner.lower() == "rrt":
            planner = RRT(pstart, ptarget, pshape, 0.1, 0.03, 0.1, lambda q: configCollidesWithBoxes(q, segments, boxes))
        elif args.planner.lower() == "prm":
            planner = PRM(pstart, ptarget, pshape, 0.01, lambda q: configCollidesWithBoxes(q, segments, boxes))
        else:
            planner = PRM(pstart, ptarget, pshape, 0.01, lambda q: configCollidesWithBoxes(q, segments, boxes))

    # execute path
    if key == ord('e'):
        if isinstance(planner, Planner):
            if planner.finished:
                path = planner.path()

ui.close()
