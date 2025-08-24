import os
import xml.etree.ElementTree as ET

# 📂 Paths
xml_dir = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\xmlval"
txt_dir = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\txtval"
class_file = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\classname.txt"

os.makedirs(txt_dir, exist_ok=True)

# ✅ Fixed class list (order matters!)
all_classes = [
    'Bollworm',
    'Meadow borer',
    'Gryllotalpa orientalis',
    'Agriotes fuscicollis Miwa',
    'Nematode trench',
    'Athetis lepigone',
    'Little Gecko',
    'Scotogramma trifolii Rottemberg',
    'Armyworm',
    'Spodoptera cabbage',
    'Anomala corpulenta',
    'Spodoptera exigua',
    'Plutella xylostella',
    'holotrichia parallela',
    'Rice planthopper',
    'Land tiger',
    'Yellow tiger',
    'eight-character tiger',
    'holotrichia oblita',
    'Stem borer',
    'Striped rice bore',
    'Spodoptera litura',
    'Rice Leaf Roller',
    'Melahotus'
]

def convert(size, box):
    """Convert VOC box to YOLO format (normalized)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0   # center x
    y = (box[2] + box[3]) / 2.0   # center y
    w = box[1] - box[0]           # width
    h = box[3] - box[2]           # height
    return (x * dw, y * dh, w * dw, h * dh)

for file in os.listdir(xml_dir):
    if not file.endswith(".xml"):
        continue

    in_file = os.path.join(xml_dir, file)
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_file = os.path.join(txt_dir, file.replace(".xml", ".txt"))
    with open(txt_file, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text.strip()

            if cls not in all_classes:
                print(f"⚠️ Warning: class '{cls}' not in predefined list, skipping...")
                continue

            cls_id = all_classes.index(cls)

            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")

# Save fixed class names to file
with open(class_file, "w", encoding="utf-8") as f:
    for c in all_classes:
        f.write(c + "\n")

print("✅ Conversion done.")
print(f"📄 Class names saved to {class_file}")
