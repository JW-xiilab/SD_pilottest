import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

clicked_points = []
clone = None
def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))

img = Image.open('/DATA_17/kjw/SD_pilottest/FloodNetV2/test/austin16-11-_png_jpg.rf.1c677db4c8a8d80f34e89c532636dd3c.jpg')
np_img = np.array(img)
opencv_image=cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
plt.imshow(opencv_image)
plt.show()