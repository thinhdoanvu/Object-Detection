'''
DICOM (.dcm)
   ↓
Read + windowing + normalize
   ↓
Convert → JPG / PNG
   ↓
Read annotation (bbox, class)
   ↓
Merge bbox (nếu multi-radiologist)
   ↓
Convert bbox → YOLO format
   ↓
Save image + label.txt

'''
# pip install pydicom opencv-python numpy
# pip install matplotlib

import pydicom, cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import defaultdict

'''
STEP 0. Hiển thị DICOM thử chơi
'''

def show_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    if hasattr(ds, "RescaleSlope"):
        img = img * ds.RescaleSlope + ds.RescaleIntercept

    # normalize
    img = (img - img.min()) / (img.max() - img.min())

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title("DICOM preview")
    plt.axis("off")
    plt.show()

    return img


'''
STEP 1. Đọc DICOM + windowing
'''

def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # HU transform (CT / CXR compatible)
    if hasattr(ds, "RescaleSlope"):
        img = img * ds.RescaleSlope + ds.RescaleIntercept

    # normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    return img

'''
STEP 2. CONVERT .DICOM to JPG
'''

def convert_dicom(soure_img, destination_img):
    # Chuyển train
    print(f"Reading DICOM images from {soure_img} folder...")
    for im in tqdm.tqdm(os.listdir(soure_img)):
        if not im.endswith(".dicom"):
            continue
        im_path = os.path.join(soure_img, im)
        img = read_dicom(im_path)
        out_name = os.path.splitext(im)[0] + ".jpg"  # đổi đuôi JPG
        cv2.imwrite(os.path.join(destination_img, out_name), img)
    print("Done!")


'''
STEP 3. CONVERT .DICOM to YOLO
 - Đọc train.csv
    + Với mỗi image_id:
    + Đọc ảnh tương ứng (JPG đã convert từ DICOM) → lấy width × height
    + Union tất cả bbox (nhiều rad_id)
    + Convert sang YOLO format
    + Chỉ 1 class duy nhất: Cardiomegaly (class_id = 0)
    + Lưu vào labels/train/{image_id}.txt
'''

# Hàm union bbox
def union_boxes(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2

# Convert bbox → YOLO format
def to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h

def convert_label_2_yolo(source_img, source_ann, destination_lbl): # đường dẫn thư mục ảnh JPG

    # 1. Đọc và ghi nhận CLASS ID
    print(f"Reading labels from {source_ann} folder...")
    # 1. Đọc CSV
    df = pd.read_csv(source_ann)
    class_names = sorted(df["class_name"].unique())
    CLASS_2_ID = {name: idx for idx, name in enumerate(class_names)}

    print("Class mapping:")
    for k, v in CLASS_2_ID.items():
        print(v, k)

    # lưu classes.txt (mỗi dòng 1 class)
    with open(classes_txt_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    # 2️. Gom bbox theo image_id
    boxes_by_image = defaultdict(list)

    for _, row in df.iterrows():
        image_id = row["image_id"]
        CLASS_ID = CLASS_2_ID[row["class_name"]]
        box = (
            float(row["x_min"]),
            float(row["y_min"]),
            float(row["x_max"]),
            float(row["y_max"]),
        )
        boxes_by_image[image_id].append(box)

    # 3. Xử lý từng ảnh JPG
    for image_id, boxes in tqdm.tqdm(boxes_by_image.items()):
        img_path = os.path.join(source_img, image_id + ".jpg")

        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        # đọc ảnh lấy kích thước
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        # union bbox
        merged_box = union_boxes(boxes)

        # convert YOLO
        xc, yc, bw, bh = to_yolo(merged_box, w, h)

        # ghi label
        label_path = os.path.join(destination_lbl, image_id + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print("Done!")


if __name__ == '__main__':
    # file class.txt để dùng sau này
    classes_txt_path = "../../datasets/Cardiomegaly/classes.txt"

    # thư mục nguồn DICOM
    IMG_SRC_TRAIN = "../../datasets/Cardiomegaly/train"
    IMG_SRC_VALID = "../../datasets/Cardiomegaly/val"

    # thư mục lưu ảnh JPG
    IMG_TRAIN = "../../datasets/Cardiomegaly/images/train"
    IMG_VAL = "../../datasets/Cardiomegaly/images/val"

    # nguồn DICOM ANNOTATION
    ANN_SRC_TRAIN = "../../datasets/Cardiomegaly/annotations/annotations_train.csv"
    ANN_SRC_VALID = "../../datasets/Cardiomegaly/annotations/annotations_test.csv"

    # thư mục lưu labels YOLO
    LBL_TRAIN = "../../datasets/Cardiomegaly/labels/train"
    LBL_VAL = "../../datasets/Cardiomegaly/labels/val"

    # tạo folder nếu chưa tồn tại
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(LBL_VAL, exist_ok=True)

    OUTPUT_VIDEO = None # Nếu muốn save thì nhập đường dẫn vào đây, ví dụ: ../../datasets/Cardiomegaly/video

    '''
    Thử đọc 1 ảnh dicom xem sao
    '''
    # show_dicom(r"C:\Users\VU\Documents\OBD\datasets\Cardiomegaly\val\0a6fd1c1d71ff6f9e0f0afa746e223e4.dicom")

    '''
    Lưu ảnh JPG
    '''
    #convert_dicom(IMG_SRC_TRAIN, IMG_TRAIN)
    #convert_dicom(IMG_SRC_VALID, IMG_VAL)

    '''
    Convert DICOM images to YOLO format
    '''
    # convert_label_2_yolo(IMG_TRAIN, ANN_SRC_TRAIN, LBL_TRAIN)
    # convert_label_2_yolo(IMG_VAL, ANN_SRC_VALID, LBL_VAL)

    '''
    Check annotation
    '''
    # Thực hiện gọi file visualize_annotate_onebyone.py đã được viết sẵn

