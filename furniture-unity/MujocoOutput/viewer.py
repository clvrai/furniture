import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_xml, load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import glob
import cv2
xmls = glob.glob("*.xml")
xmls = [ "0005_CHAIR_AGAM.xml"]

for xml in xmls:
    model = load_model_from_path(xml)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    #viewer = MjRenderContextOffscreen(sim, -1)
    while True:
        viewer.render()
    #step = 0
    #viewer.render(420, 380, -1)
    #data = np.asarray(viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)
    #if data is not None:
    #    cv2.imwrite(f"{xml}.png", data)
    #del viewer
