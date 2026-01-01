import pydicom
import cv2
import numpy as np
import tqdm
import os
import pandas as pd
import json
from datetime import datetime


# --- HELPER FUNCTIONS ---

def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope"):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    # Normalize về 0-255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img


def union_boxes(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


# --- MAIN CONVERTER ---

def convert_to_coco(dicom_dir, csv_path, output_jpg_dir, output_json_path):
    os.makedirs(output_jpg_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1. Khởi tạo cấu trúc COCO
    coco_format = {
        "info": {
            "description": "Medical DICOM Dataset to COCO",
            "date_created": datetime.now().isoformat()
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 2. Tạo Categories
    class_names = sorted(df["class_name"].unique())
    category_map = {}
    for i, name in enumerate(class_names):
        category_map[name] = i + 1  # COCO thường bắt đầu từ index 1
        coco_format["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "none"
        })

    # 3. Gom nhóm dữ liệu theo image_id (để xử lý union bbox nếu cần)
    grouped = df.groupby("image_id")

    ann_id_cnt = 1
    img_id_cnt = 1

    print(f"Processing {len(grouped)} unique images to COCO...")

    for image_id, group in tqdm.tqdm(grouped):
        dicom_path = os.path.join(dicom_dir, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            continue

        # Đọc DICOM và lưu JPG
        img_array = read_dicom(dicom_path)
        h, w = img_array.shape
        jpg_name = f"{image_id}.jpg"
        cv2.imwrite(os.path.join(output_jpg_dir, jpg_name), img_array)

        # Thêm vào phần 'images'
        coco_format["images"].append({
            "id": img_id_cnt,
            "file_name": jpg_name,
            "width": w,
            "height": h
        })

        # Xử lý Annotations (Trong ví dụ này ta gom các box cùng class của các bác sĩ lại)
        # Nếu muốn giữ riêng từng box của từng bác sĩ, bỏ qua bước union_boxes

        # Ví dụ: Union tất cả box của cùng một ảnh (vì bài toán Cardiomegaly thường chỉ có 1 vùng tim)
        boxes = group[["x_min", "y_min", "x_max", "y_max"]].values
        merged_box = union_boxes(boxes)

        x_min, y_min, x_max, y_max = merged_box
        bw = x_max - x_min
        bh = y_max - y_min
        area = bw * bh

        # Lấy class_id từ class_name đầu tiên trong nhóm
        class_name = group.iloc[0]["class_name"]
        cat_id = category_map[class_name]

        coco_format["annotations"].append({
            "id": ann_id_cnt,
            "image_id": img_id_cnt,
            "category_id": cat_id,
            "bbox": [x_min, y_min, bw, bh],  # COCO format: [x, y, width, height]
            "area": float(area),
            "iscrowd": 0
        })

        ann_id_cnt += 1
        img_id_cnt += 1

    # 4. Lưu file JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"Successfully saved COCO JSON to: {output_json_path}")


if __name__ == '__main__':
    # Cấu hình đường dẫn
    IMG_SRC_TRAIN = "../../datasets/Cardiomegaly/train"
    ANN_SRC_TRAIN = "../../datasets/Cardiomegaly/annotations/annotations_train.csv"
    IMG_SRC_VAL = "../../datasets/Cardiomegaly/val"
    ANN_SRC_VAL = "../../datasets/Cardiomegaly/annotations/annotations_test.csv"

    OUT_IMG_TRAIN = "../../datasets/Cardiomegaly/coco/images/train"
    OUT_JSON_TRAIN = "../../datasets/Cardiomegaly/coco/annotations/train.json"
    OUT_IMG_VAL = "../../datasets/Cardiomegaly/coco/images/val"
    OUT_JSON_VAL = "../../datasets/Cardiomegaly/coco/annotations/val.json"

    os.makedirs(os.path.dirname(OUT_JSON_TRAIN), exist_ok=True)

    # Thực hiện convert cho train
    # convert_to_coco(
    #     dicom_dir=IMG_SRC_TRAIN,
    #     csv_path=ANN_SRC_TRAIN,
    #     output_jpg_dir=OUT_IMG_TRAIN,
    #     output_json_path=OUT_JSON_TRAIN
    # )

    # Thực hiện convert cho val
    convert_to_coco(
        dicom_dir=IMG_SRC_VAL,
        csv_path=ANN_SRC_VAL,
        output_jpg_dir=OUT_IMG_VAL,
        output_json_path=OUT_JSON_VAL
    )