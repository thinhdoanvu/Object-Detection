import ultralytics
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import torchvision.transforms as transforms
from PIL import Image
import io

plt.rcParams["figure.figsize"] = [3.0, 3.0]

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

# For the object detection model
# model = YOLO('models/ip102_Finalv2l_691.pt')
# model = YOLO('models/ip102_yolov8m_67.8.pt')
# model = YOLO('models/ip102_yolov8s_68.3.pt')
# model = YOLO('models/ip102_yolov9m_687.pt')
# model = YOLO('models/ip102_yolov10l_625.pt')
# model = YOLO('models/yolo11n.pt')
# model = YOLO('models/ip102_yolov8x_69.7.pt')
model = YOLO('models/k4_69.4.pt')
# model = YOLO('models/k5_69.3.pt')
# model = YOLO('models/r2000_81.1.pt')
# model = YOLO('models/pest24_v9m.pt')
# model = YOLO('models/biodetr.pt')   # activate biodetr first
# model = YOLO('models/rtdetr.pt')   # activate thinhdv first


model.cpu()

img = cv2.imread('images/IP038000453.jpg')
# img = cv2.imread('images/IP014000284.jpg')
# img = cv2.imread('images/00_00046_.jpg')

img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255

target_layers =[model.model.model[-4]]

cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
plt.imshow(cam_image)
plt.savefig(f'outputs/k4.jpg')  # Save the plot as an image file
plt.show()
plt.close()

