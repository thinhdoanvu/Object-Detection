import os
import json
import xml.etree.ElementTree as ET

# 📂 Paths
xml_dir = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\xmltrain"
json_file = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\annotations_train.json"

# Containers
images = []
annotations = []
categories = []
category_set = {}

ann_id = 0
img_id = 0

for file in os.listdir(xml_dir):
    if not file.endswith(".xml"):
        continue

    in_file = os.path.join(xml_dir, file)
    tree = ET.parse(in_file)
    root = tree.getroot()

    # image info
    filename = root.find("filename").text
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    images.append({
        "id": img_id,
        "file_name": filename,
        "width": w,
        "height": h
    })

    # annotations
    for obj in root.iter("object"):
        cls = obj.find("name").text.strip()

        if cls not in category_set:
            cat_id = len(category_set)
            category_set[cls] = cat_id
            categories.append({"id": cat_id, "name": cls})

        cat_id = category_set[cls]

        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)

        w_box = xmax - xmin
        h_box = ymax - ymin
        ann = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [xmin, ymin, w_box, h_box],
            "area": w_box * h_box,
            "iscrowd": 0
        }
        annotations.append(ann)
        ann_id += 1

    img_id += 1

# Final COCO dict
coco_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(coco_dict, f, indent=4)

print("✅ Conversion done. COCO annotations saved at:", json_file)
