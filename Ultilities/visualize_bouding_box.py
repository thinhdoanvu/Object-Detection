import cv2
import os
import matplotlib.pyplot as plt
import random

# Path to the image and its corresponding annotation file
image_path = "F:/users/thanh/ntu_group/thinh/ObjectDetection/datasets/SODAD/train/images/00010.jpg"
annotation_path = "F:/users/thanh/ntu_group/thinh/ObjectDetection/datasets/SODAD/train/labels/00010.txt"
classes_path = "F:/users/thanh/ntu_group/thinh/ObjectDetection/datasets/SODAD/classes.txt"

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Load class names
with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Generate random colors for each class
colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
          for i in range(len(class_names))}

# Read the annotation file
with open(annotation_path, "r") as file:
    for line in file.readlines():
        # Each line: class_id x_center y_center width height
        parts = line.strip().split()
        class_id = int(parts[0])  # Class ID
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
        color = colors.get(class_id, (255, 255, 255))  # Default: white
        class_name = class_names[class_id] if class_id < len(class_names) else f"id_{class_id}"
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display and save the image
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()
