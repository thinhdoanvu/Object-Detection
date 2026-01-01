import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 📂 Paths
xml_file = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\xmltest\0000003.xml"
img_dir = r"F:\users\thanh\ntu_group\thinh\ObjectDetection\datasets\Pest24v2\voc2007\voc2007\images"

# --- Parse XML ---
tree = ET.parse(xml_file)
root = tree.getroot()

# Get image file name from <filename>
img_filename = root.find("filename").text
if not img_filename.lower().endswith(".jpg"):
    img_filename += ".jpg"

img_path = os.path.join(img_dir, img_filename)

# Load image
img = Image.open(img_path).convert("RGB")
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(img)

# Draw each bounding box
for obj in root.findall("object"):
    cls = obj.find("name").text
    xmlbox = obj.find("bndbox")
    xmin = int(xmlbox.find("xmin").text)
    ymin = int(xmlbox.find("ymin").text)
    xmax = int(xmlbox.find("xmax").text)
    ymax = int(xmlbox.find("ymax").text)

    # Draw rectangle
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)
    # Draw class name
    ax.text(xmin, ymin - 5, cls, color="yellow", fontsize=10, backgroundcolor="black")

plt.axis("off")
plt.show()
