import torch
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sys
import logging
import os

sys.path.insert(0, os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], 'yolov5'))

from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox


class YOLO_v5_OE():
    '''
    Class for storing a YOLOv5 model instance

    Use the "predict" function to infere the location of the OE logos
    '''

    def __init__(self, weights_pt_path):
        '''
        Constructor of YOLO_v5_OE class

        arguments:
         - weights_pt_path (string): Path to the .pt file containing the weight of the model
        '''
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.WARNING)
        matplotlib.use('TkAgg')
        self.device = select_device('')
        self.model = DetectMultiBackend(weights_pt_path, device=self.device, dnn=False)
        self.log.info('Loaded model!')

    def set_loglevel(self, level=logging.WARNING):
        self.log.setLevel(level)


    def predict(self, input_img):
        '''
        Infere the location of the OE logo on the image, with the highest prediction confidence

        arguments:
         - input_img (np.array): The image in BGR format as a numpy array (from cv2.imread())

        returns:
         - detection_result (tuple): (u,v) image coordinates of the centerpoint of the "O" of the best detected OE logo
        '''
        # Get greater dimension of image
        img_larger_shape = input_img.shape[1] if input_img.shape[1] >= input_img.shape[0] else input_img.shape[0]

        # Pad the image
        padded = letterbox(input_img, img_larger_shape, self.model.stride, self.model.pt and not self.model.jit)[0]
        padded = padded.transpose((2, 0, 1))[::-1]
        padded = np.ascontiguousarray(padded)

        if len(padded.shape) == 3:
            # Add batch dimension if there is none
            padded = padded[None]

        # Create Torch tensor from padded image
        img = torch.from_numpy(padded).to(self.device)
        img = img.float()
        img /= 255

        # Predict with the model
        pred = self.model(img)
        pred = non_max_suppression(pred)

        # Visualize image
        if self.log.level < logging.WARNING:
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        detection_result = []

        # Get x,y,w,h values from predictions
        for i, det in enumerate(pred):
            gn = torch.tensor(input_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_img.shape).round()
                for *xyxy, conf, classif in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    x_min = (xywh[0]-xywh[2]/2)*input_img.shape[1]
                    y_min = (xywh[1]-xywh[3]/2)*input_img.shape[0]
                    w = xywh[2]*input_img.shape[1]
                    h = xywh[3]*input_img.shape[0]

                    detection_result.append((conf.item(), int(classif.item()), (x_min,y_min,w,h)))

                    if self.log.level < logging.WARNING:
                        rect = patches.Rectangle((x_min,y_min), w,h, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
        if self.log.level < logging.WARNING:
            plt.show()

        if detection_result:
            # There are detected logos, select the one with the highest confidence
            detection_result.sort()
            a = list(reversed(detection_result))
            detected_obj = a[0]

            xywh = detected_obj[2]  # Get xywh of highest-confidence logo

            # Crop image to mounding-box and turn it into grayscale
            cropped = input_img[int(xywh[1]):int(xywh[1]+xywh[3]),int(xywh[0]):int(xywh[0]+xywh[2]),:]
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray_cropped = cv2.medianBlur(gray_cropped,5)

            # Get smaller dimension of bounding box
            cropped_smaller_dim = cropped.shape[0] if cropped.shape[0]<= cropped.shape[1] else cropped.shape[1]

            # Predict middlepoint of 'O' with HoughCircles
            circles = cv2.HoughCircles(gray_cropped,cv2.HOUGH_GRADIENT,1,int(cropped_smaller_dim/4),param1=50,param2=30,minRadius=2,maxRadius=int(cropped_smaller_dim/4))

            if circles is None:
                # If no circles were found allow finding the outer circle
                circles = cv2.HoughCircles(gray_cropped,cv2.HOUGH_GRADIENT,1,int(cropped_smaller_dim/2),param1=50,param2=30,minRadius=2,maxRadius=int(cropped_smaller_dim/2))

            if self.log.level < logging.WARNING:
                fig, ax = plt.subplots()
                ax.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            if not circles is None:
                circles = np.uint16(np.around(circles))
                if self.log.level < logging.WARNING:
                    for i in circles[0,:]:
                        circ = patches.Circle((i[0],i[1]), i[2], linewidth=5, edgecolor='g', facecolor='none')
                        circ_mid = patches.Circle((i[0],i[1]), 1, linewidth=5, edgecolor='r', facecolor='r')
                        ax.add_patch(circ)
                        ax.add_patch(circ_mid)

                    plt.show()
                return(xywh[0] + circles[0][0][0], xywh[1] + circles[0][0][1])
            else:
                self.log.warn('No circles found in the best-confidence bounding box!')
                return None
        else:
            self.log.warn('No OE logos found on the input image')
            return None



if __name__=='__main__':
    model = YOLO_v5_OE('../../oe_logo_demo/best.pt')
    img_np = cv2.imread('../../oe_logo_demo/real1.png')

    model.set_loglevel(logging.WARNING)

    print(model.predict(img_np))
