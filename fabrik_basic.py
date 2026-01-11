import numpy as np
import cv2 as cv
import matrix as mat
import ui



wsize = (800, 800)
base = np.array(wsize) / 2
segments = [
    (0, 110),
    (0, 300),
    (0, 150),
    (0, 120),
    (0, 90)
]

target = np.array([0, 0])
mouseDown = False
def on_mouse(event, x, y, flags, param):
    global target, mouseDown
    if event == cv.EVENT_LBUTTONDOWN:
        mouseDown = True
    if event == cv.EVENT_MOUSEMOVE and mouseDown:
        target = np.array([x,y])
    if event == cv.EVENT_LBUTTONUP:
        mouseDown = False

canvas = ui.new("scene", wsize, on_mouse)

while ui.show(20):
    segments = mat.FABRIK_step(target, base, segments)
    canvas &= 0
    canvas = ui.drawChain(canvas, base, segments, target)

ui.close()
