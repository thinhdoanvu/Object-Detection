import ultralytics
from ultralytics import YOLO
from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

# Load YOLO model
# model = YOLO('models/yolo11l.pt')
model = RTDETR('models/rtdetr.pt') # # activate thinhdv first
# model = YOLO('models/r2000_81.1.pt')# # activate thinhdv first

model.cpu()

# Load image
img = cv2.imread('images/14_00135_.jpg')
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255.0

# Setup target layer
target_layers = [model.model.model[-4]]

# Run EigenCAM
cam = EigenCAM(model, target_layers, task='od')
grayscale_cam = cam(rgb_img)[0, :, :]

# Show CAM overlay on image
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# Plot with colorbar
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cam_image)
ax.axis('off')

# Add colorbar with range 0–255
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label('CAM Intensity (0–255)')
cbar.set_ticks([0, 64, 128, 192, 255])
cbar.set_ticklabels(['0', '64', '128', '192', '255'])

# Save and show
plt.savefig('outputs/rtdetr.jpg', bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
