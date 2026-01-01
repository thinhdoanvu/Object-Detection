import cv2
import os
import matplotlib.pyplot as plt

# Path to the image
# image_path = r"C:\Users\VU\Documents\OBD\datasets\IP102\train\images\IP004000097.jpg"
image_path = r"C:\Users\VU\Documents\OBD\Ultility\IP004000097_o.jpg"

# Load the image
image = cv2.imread(image_path)

# Resize ảnh xuống nếu quá lớn (giữ tỷ lệ)
max_width = 500
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    new_size = (max_width, int(image.shape[0] * scale))
    image = cv2.resize(image, new_size)

# Input values: class_id, confidence, xmin, ymin, xmax, ymax
class_id, confidence, xmin, ymin, xmax, ymax = 20, 0.74, 280, 25, 480, 260

# Màu cố định cho bounding box (BGR)
fixed_color = (67, 203, 90)

# Gán nhãn: class_id + confidence score
class_name = f"{class_id:02d} {confidence:.2f}"

# Vẽ bounding box
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), fixed_color, 2)

# --- Vẽ nền xanh + chữ trắng cho nhãn ---
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

# Tính kích thước chữ
(text_w, text_h), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)

# Tọa độ góc dưới trái của chữ
text_x = xmin
text_y = ymin - 10 if ymin - 10 > text_h else ymin + text_h + 10

# Vẽ nền xanh (rectangle)
cv2.rectangle(image, (text_x, text_y - text_h - baseline),
              (text_x + text_w, text_y + baseline),
              fixed_color, -1)   # nền xanh (BGR)

# Vẽ chữ trắng
cv2.putText(image, class_name, (text_x, text_y),
            font, font_scale, (255, 255, 255), thickness)

# Convert BGR → RGB để hiển thị bằng matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Xuất file output dựa trên tên ảnh gốc
output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_o.jpg"

plt.imshow(image_rgb)
plt.axis('off')
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
