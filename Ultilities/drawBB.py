import cv2
import os
import matplotlib.pyplot as plt

# Path to the image
image_path = r"C:\Users\VU\Documents\OBD\datasets\IP102\train\images\IP004000097.jpg"

# Load the image
image = cv2.imread(image_path)

# Resize ảnh xuống nếu quá lớn (giữ tỷ lệ)
max_width = 500
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    new_size = (max_width, int(image.shape[0] * scale))
    image = cv2.resize(image, new_size)

# Input values: class_id, confidence, xmin, ymin, xmax, ymax
class_id, confidence, xmin, ymin, xmax, ymax = 20, 0.84, 12, 28, 480, 290

# Màu cố định cho mọi class_id (BGR)
fixed_color = (67, 203, 90)

# Gán nhãn: class_id + confidence score
class_name = f"{class_id:02d} {confidence:.2f}"

# Vẽ bounding box và label
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), fixed_color, 2)
cv2.putText(image, class_name, (xmin, ymin - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,   fixed_color,1)

# Convert BGR → RGB để hiển thị bằng matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Xuất file output dựa trên tên ảnh gốc
output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_o.jpg"

plt.imshow(image_rgb)
plt.axis('off')
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
