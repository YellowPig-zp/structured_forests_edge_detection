import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

model = "model.yml"
im = "02.jpg"
im = cv.imread(im)

edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
print(type(rgb_im))
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

orimap = edge_detection.computeOrientation(edges)
edges = edge_detection.edgesNms(edges, orimap)


cv.imshow("edges", edges)
cv.imwrite("1.jpg", edges*255.0)
cv.waitKey(0)