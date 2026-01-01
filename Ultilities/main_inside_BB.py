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
from torchvision.ops import roi_align

# Load YOLO model
# model = YOLO('models/yolov8n.pt')#bio
# model = YOLO('models/yolov8s.pt') #rt
# model = YOLO('models/yolov8l.pt')
# model = YOLO('models/yolov9c.pt')
# model = YOLO('models/yolov9m.pt')#hsdpa
# model = YOLO('models/yolov10l.pt')
# model = YOLO('models/k4_69.4.pt')
# model = YOLO('models/yolo11l.pt')
# model = YOLO('models/yolov9s.pt') #12
# model = RTDETR('models/rtdetr.pt') # # activate thinhdv first
# model = YOLO('models/r2000_81.1.pt')# # activate thinhdv first
# model = YOLO('models/biodetr.pt')# activate biodetr first
# model = YOLO('models/yolo12_r2k_79.1.pt') # activate yolo12 first

# model = YOLO(r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\9m_l2\weights\best.pt") #attention on BB
model = YOLO(r"C:\Users\VU\Documents\OBD\v3_H200_backup\v4\v4_960_95.5_67.7\weights\best.pt")
# model = YOLO(r"C:\Users\VU\Documents\OBD\yolov12\runs\detect\chuan\weights\best.pt") #WAFU

model.cpu()

# Load image
images = [
    # 'C:/Users/VU/Documents/OBD/datasets/MangoPests/train/images/beetle (1).jpg',
    # 'C:/Users/VU/Documents/OBD/AICUP25/train/images/patient0001_0009.png',
    'C:/Users/VU/Documents/OBD/AICUP25/train/images/patient0001_0229.png',
    # 'C:/Users/VU/Documents/OBD/AICUP25/test/images/patient0051_0139.png',
    'C:/Users/VU/Documents/OBD/AICUP25/test/images/patient0051_0227.png',
    'C:/Users/VU/Documents/OBD/AICUP25/test/images/patient0051_0259.png'
]

for image_path in images:
    img = cv2.imread(image_path)
    if img is None: continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img) / 255.0

    results = model(image_path)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    target_layers = [model.model.model[-4]]
    cam_gen = EigenCAM(model, target_layers, task='od')
    grayscale_cam = cam_gen(rgb_img)[0, :, :]

    # 1. Tạo Distance Map
    # Khởi tạo mask đen, vùng trong box là trắng (1)
    binary_mask = np.zeros_like(grayscale_cam, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        binary_mask[y1:y2, x1:x2] = 1

    # Tính khoảng cách từ mỗi pixel tới Box gần nhất
    # cv2.DIST_L2 là khoảng cách đường chim bay
    dist_map = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 3)

    # 2. Tạo hàm suy giảm (Attenuation Function)
    # Càng xa box, giá trị càng giảm. Sau 100 pixel thì cường độ chỉ còn 10%
    decay_constant = 200  # Bạn có thể tăng/giảm số này để chỉnh độ rộng vùng tập trung
    soft_attenuation = np.exp(-dist_map / decay_constant)

    # 3. Kết hợp với CAM gốc
    # Vùng trong Box (dist=0) -> exp(0) = 1 (giữ nguyên)
    # Vùng rìa ảnh (dist lớn) -> exp(-dist) tiến về 0 (đỏ thành xanh)
    refined_cam = grayscale_cam * soft_attenuation

    # 4. Hiển thị
    cam_image = show_cam_on_image(img_float, refined_cam, use_rgb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cam_image)
    ax.axis('off')
    plt.title("Bio-realistic Attenuated CAM")
    plt.show()
    plt.close()
