import cv2
import os
import matplotlib.pyplot as plt

# Path to the image and its corresponding annotation file
image_path = r"C:\Users\VU\Documents\OBD\datasets\R2000\images\classID_01\01_00116_.jpg"
annotation_path = r"C:\Users\VU\Documents\OBD\datasets\R2000\labels\classID_01\01_00116_.txt"

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define colors for bounding boxes
colors = {
    1: (230, 217, 17),  # bounding box màu vàng cho class 1
}

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

        # Draw the rectangle
        color = colors.get(class_id, (230, 217, 17))  # Default color
        class_name = f"{class_id:02d}"  # Label text

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

        # --- Vẽ nền + chữ ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
        text_x = xmin
        text_y = ymin - 5 if ymin - 5 > text_h else ymin + text_h + 5

        # Vẽ nền (màu theo mã hex #11D9E6 → BGR (230,217,17))
        cv2.rectangle(image,
                      (text_x, text_y - text_h - baseline),
                      (text_x + text_w, text_y + baseline),
                      (230, 217, 17), -1)

        # Vẽ chữ trắng
        cv2.putText(image, class_name, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display and save the image
plt.imshow(image_rgb)
plt.axis('off')
plt.savefig('01_00116_gt.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
