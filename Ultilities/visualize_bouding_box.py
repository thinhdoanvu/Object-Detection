import cv2
import os

# Path to the image and its corresponding annotation file
image_path = "E:/thanh/ntu_group/thinh/ObjectDetection/datasets/BloodCell/BCCD/train/images/BloodImage_00001.jpg"
annotation_path = "E:/thanh/ntu_group/thinh/ObjectDetection/datasets/BloodCell/BCCD/train/labels/BloodImage_00001.txt"

# Load the image
print(os.path.exists("E:/thanh/ntu_group/thinh/ObjectDetection/datasets/BloodCell/BCCD/train/labels/BloodImage_00001.txt"))
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define colors for different class IDs
colors = {
    0: (0, 255, 0),  # Green for class 0
    1: (0, 0, 255),  # Red for class 1
    2: (255, 0, 0)   # Blue for class 2
}

# Read the annotation file
with open(annotation_path, "r") as file:
    for line in file.readlines():
        # Each line: class_id x_center y_center width height
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        # Convert YOLO format to bounding box coordinates
        xmin = int(x_center - box_width / 2)
        ymin = int(y_center - box_height / 2)
        xmax = int(x_center + box_width / 2)
        ymax = int(y_center + box_height / 2)

        # Draw the rectangle and label on the image
        color = colors.get(class_id, (255, 255, 255))  # Default color: white
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, f"Class {class_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

import matplotlib.pyplot as plt

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.title("Labeled Image")
plt.show()