# import cifar10_loader
# import BSR_loader
import ImageNet_loader
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing 

batch_names = ["data_batch_{}".format(i+1) for i in range(5)]
model = "model.yml"
ImageNet_directory = "/home/shared/rodia/datasets/imagenet/train_256x256/"
# ImageNet_directory = "./ImageNet/"

def edge_detect(img):
    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    edges = edge_detection.detectEdges(np.float32(img))
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    return edges

class myThread(multiprocessing.Process):
    def __init__(self, thread_name, images, start_index, end_index):
      multiprocessing.Process.__init__(self)
      self.thread_name = thread_name
      self.thread_images = images[start_index:end_index]


    def run(self):
        thread_edge_maps_batch = []
        for i in range(len(self.thread_images)):
            img = self.thread_images[i]
            edge_map = edge_detect(img)
            thread_edge_maps_batch.append(edge_map)
            if i%20==0:
                print(i, len(self.thread_images), self.thread_name)
        self._return = thread_edge_maps_batch

    def join(self):
        multiprocessing.Process.join(self)
        return self._return



if __name__ == "__main__":
    # name = batch_names[0]
    for batch_name in os.listdir(ImageNet_directory):
        if batch_name in os.listdir("./ImageNet_edge_maps/"):
            print("{} removed!".format(batch_name))
            continue
        print("{} starts:".format(batch_name))
        full_filename = ImageNet_directory + batch_name
        images = ImageNet_loader.load(full_filename)
        # images, labels = BSR_loader.load()
        edge_maps_batch = []
        num_size = len(images)

        num_threads = 8

        my_threads = []
        for i in range(num_threads):
            if not i == num_threads - 1:
                tmp_thread = myThread("Thread-{}".format(i+1), \
                    images, num_size//num_threads*i, num_size//num_threads*(i+1))
            else:
                tmp_thread = myThread("Thread-{}".format(i+1), \
                    images, num_size//num_threads*i, num_size)
            my_threads.append(tmp_thread)

        for idx, t in enumerate(my_threads):
            os.system("taskset -p -c %d %d" % (idx % multiprocessing.cpu_count(), os.getpid()))
            t.start()

        my_thread_results = []
        for t in my_threads:
            my_thread_results.append(t.join())

        print("Exiting Main Thread")
        # thread1 = myThread("Thread-1", images[:num_size//num_threads])
        # thread2 = myThread("Thread-2", images[num_size//num_threads:num_size//num_threads*2])
        # thread3 = myThread("Thread-3", images[num_size//num_threads*2:num_size//num_threads*3])
        # thread4 = myThread("Thread-4", images[num_size//num_threads*3:])
        # thread5 = myThread("Thread-5", images[num_size//8*4:num_size//8*5])
        # thread6 = myThread("Thread-6", images[num_size//8*5:num_size//8*6])
        # thread7 = myThread("Thread-7", images[num_size//8*6:num_size//8*7])
        # thread8 = myThread("Thread-8", images[num_size//8*7:])

        # thread1.start()
        # thread2.start()
        # thread3.start()
        # thread4.start()
        # thread5.start()
        # thread6.start()
        # thread7.start()
        # thread8.start()

        # batch1 = thread1.join()
        # batch2 = thread2.join()
        # batch3 = thread3.join()
        # batch4 = thread4.join()
        # batch5 = thread5.join()
        # batch6 = thread6.join()
        # batch7 = thread7.join()
        # batch8 = thread8.join()

        # print ("Exiting Main Thread")
        
        edge_maps_batch = []
        for result in my_thread_results:
            edge_maps_batch += result
        edge_maps_batch = np.array(edge_maps_batch)
        np.save("./ImageNet_edge_maps/"+batch_name, edge_maps_batch)

        