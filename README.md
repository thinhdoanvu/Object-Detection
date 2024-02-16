# Object-Detection-By-YOLOv8
## Bước 1. Gán nhãn cho ảnh
### Truy cập: https://app.supervisely.com/
### Tiến hành gán nhãn
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/5f3a2b6c-61d3-41e7-a3fa-ce7215a3821d)

## Bước 2. Tải label với định dạng YOLOV5 (YOLOV8 chỉ cho định dạng line, polygon...)
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/d67aec3e-f678-4a7c-97c2-b3abbc79903b)

### Kết quả ta có cấu trúc thư mục như sau:
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/5641ba4b-a8db-4e51-b367-0e24b28cc91b)

## LƯU Ý: 
KHI THỰC HIỆN TRAIN MODEL, YOLO ĐÒI HỎI PHẢI CÓ ẢNH VÀ LABEL TRONG FOLDER val. DO ĐÓ, TRONG THỰC TẾ, supervisely SẼ TỰ PHÂN CHIA. TUY NHIÊN CHÚNG TA NÊN TRÍCH NGẪU NHIÊN TRONG FOLDER TRAIN (TỐI THIỂU 10%).

## THỰC HIỆN TRÊN GOOGLE COLAB:
### Chuẩn bị data: mount folder google drive
from google.colab import drive
drive.mount('/content/drive')

### Chuẩn bị data: tải yolov8
%%bash

cd /content/drive/MyDrive/

mkdir -p DemHongCau

cd DemHongCau

git clone https://github.com/ultralytics/ultralytics.git

mv ultralytics yolov8

ls

### Chuẩn bị data: setup môi trường
%pip install ultralytics

import ultralytics

ultralytics.checks()

#### Kết quả: 
Ultralytics YOLOv8.1.14 🚀 Python-3.10.12 torch-2.1.0+cu121 CPU (Intel Xeon 2.20GHz)

Setup complete ✅ (2 CPUs, 12.7 GB RAM, 26.4/107.7 GB disk)

### Thiết lập cấu hình
path: /content/drive/MyDrive/DemHongCau/yolov8/data

train: images/train

val: images/val

nc: 1

names: ['hongcau']

## LƯU Ý:
Lưu ý: cần copy ít nhất 1 ảnh (hình + label) vào thư mục val yolov8n.pt nghĩa là file pretrain weight của yolo nornal (khác với extra, tiny...)
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/3e9a2e36-9130-494f-a83f-43599065895a)

### Train: chỉnh sửa file config.yaml
path: /content/drive/MyDrive/DemHongCau/yolov8/data

train: images/train

val: images/val

nc: 1

names: ['hongcau']

### Train
%cd /content/drive/MyDrive/DemHongCau/yolov8/data

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model.train(data="hongcau_config.yaml", epochs=100)  # train the model

metrics = model.val()  # evaluate model performance on the validation set

#### Lưu ý:
model = YOLO("yolov8n.pt") tương ứng với phiên bản của YOLOv8
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/423a2a02-add7-4383-9079-0df072546368)


#### Kết quả:
![image](https://github.com/thinhdoanvu/Object-Detection/assets/22977443/82cb22a2-908c-4506-b2c7-14d94867aac7)
