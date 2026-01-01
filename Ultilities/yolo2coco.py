import os
import json
import logging
from pylabel import importer

# Giảm bớt warning không cần thiết
logging.getLogger().setLevel(logging.CRITICAL)

# === Config ===
# Chọn folder annotations/images
path_to_annotations = "C:/Users/VU/Documents/OBD/AICUP25/val/labels"
path_to_images = "C:/Users/VU/Documents/OBD/AICUP25/val/images"
# path_to_annotations = "C:/Users/VU/Documents/OBD/AICUP25/train/labels"
# path_to_images = "C:/Users/VU/Documents/OBD/AICUP25/train/images"

yoloclasses = ['aortic_valve']  # chỉ có 1 class

# === Fix function: ép class_id về int ===
def fix_labels(path_to_annotations):
    print("🔧 Đang xử lý lại nhãn YOLO...")
    for file in os.listdir(path_to_annotations):
        if file.endswith(".txt"):
            fixed_lines = []
            with open(os.path.join(path_to_annotations, file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # ép class_id thành int (0.000 -> 0)
                        class_id = str(int(float(parts[0])))
                        fixed_line = " ".join([class_id] + parts[1:])
                        fixed_lines.append(fixed_line)
                    else:
                        print(f"⚠️ File {file} có dòng sai format: {line.strip()}")
            # overwrite lại file
            with open(os.path.join(path_to_annotations, file), "w") as f:
                f.write("\n".join(fixed_lines))

# === Bổ sung info/licenses vào COCO JSON ===
def add_info_to_coco(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nếu chưa có thì thêm
    if "info" not in data:
        data["info"] = {
            "description": "AICUP25 Dataset",
            "version": "1.0",
            "year": 2025
        }
    if "licenses" not in data:
        data["licenses"] = []

    # overwrite lại
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("✅ Đã thêm 'info' và 'licenses' vào COCO JSON.")

# === Run ===
if __name__ == "__main__":
    # B1: fix labels trước
    fix_labels(path_to_annotations)

    # B2: Import vào pylabel
    dataset = importer.ImportYoloV5(
        path=path_to_annotations,
        path_to_images=path_to_images,
        cat_names=yoloclasses,
        img_ext="png",  # chỉnh nếu ảnh là jpg
        name="_annotations.coco"
    )

    # B3: Thông tin dataset
    print(f"📂 Number of images: {dataset.analyze.num_images}")
    print(f"📂 Number of classes: {dataset.analyze.num_classes}")
    print(f"📂 Classes: {dataset.analyze.classes}")
    print(f"📂 Class counts:\n{dataset.analyze.class_counts}")

    # B4: Export ra COCO JSON
    dataset.df["cat_id"] = 0
    dataset.df["cat_name"] = "aortic_valve"
    coco_paths = dataset.export.ExportToCoco(cat_id_index=0)
    print("✅ Export COCO thành công!")

    # B5: Thêm info/licenses để RFDETR không bị lỗi
    for coco_path in coco_paths:  # coco_paths là list
        add_info_to_coco(coco_path)

