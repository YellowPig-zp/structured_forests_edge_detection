from cifar10_loader import *
import cv2 as cv
import numpy as np

batch_names = ["data_batch_{}".format(i+1) for i in range(5)]
model = "model.yml"

def edge_detect(img):
    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    edges = edge_detection.detectEdges(np.float32(img) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    return edges

if __name__ == "__main__":
    name = batch_names[0]
    images, labels = load(name)
    for i in range(len(images)):
        img = images[i]
        edges = edge_detect(img)
        cv.imwrite("./edge_maps/data_batch_1_{}.jpg".format(i), edges * 255.0)


        
