
import matplotlib.pylab as plt
import cv2
import numpy as np


class Detector:
    
    #@staticmethod
    image = cv2.imread('img/Udacity/image001.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    segmentierung = [(0, height),(width/2, height/2),(width, height)]
    def detectLines(img, vertices):
   
        mask = np.zeros_like(img)
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
           

        

        # TODO: Implement
        # Segment the image
        #image = cv.rectangle(image, (200, 200), (300, 300), (0, 0, 255), 1)
        return masked_image



    seg_image = detectLines(image, np.array([segmentierung], np.int32),)

    plt.imshow(seg_image)
    plt.show()

    # @staticmethod
    def detectObjects(image):
            """
            This method detects other objects (cars, etc.) in the given image.
            Args:
                image: The image to analyse and detect objects in
            Returns:
                The image with the detected objects
            """

            # TODO: Implement
            # https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
            # https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
            return cv2.rectangle(image, (225, 225), (275, 275), (255, 0, 0), 1)