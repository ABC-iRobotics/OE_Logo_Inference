# OE Logo Inference
Contains code for making prediction with the trained YOLO model.

## Usage
In order to use the code you need to install the [yolov5](https://github.com/ultralytics/yolov5/) python package.

## YOLO_v5_OE
The predict.py module defines a YOLO_v5_OE class that can be used to load and prediction with the YOLO model.

Example:
```python
model = YOLO_v5_OE('WEIGHT_FILE_PATH')    # create model from weight file
img_np = cv2.imread('IMAGE_FILE_PATH')    # get an image as a numpy array in openCV format
print(model.predict(img_np))    # predict with model
```