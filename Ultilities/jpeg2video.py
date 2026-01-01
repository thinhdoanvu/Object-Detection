import cv2
import os
from glob import glob

# Đường dẫn tới thư mục chứa ảnh
image_folder = '../yolo11/ultralytics/runs/detect/predict_r2k_hspda'
image_paths = sorted(glob(os.path.join(image_folder, '00_*.jpg')))
output = "class00.mp4"

# Đảm bảo có ít nhất 2 ảnh
if len(image_paths) < 2:
    raise ValueError("Không đủ ảnh để tạo video.")

# Đọc ảnh đầu tiên để lấy kích thước
frame = cv2.imread(image_paths[0])
height, width, layers = frame.shape
size = (width, height)

# Khởi tạo video writer
out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 2, size)

# Thêm từng ảnh vào video
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Không thể đọc ảnh: {path}")
        continue
    resized = cv2.resize(img, size)  # đảm bảo ảnh đúng kích thước
    out.write(resized)
    print(f"✅ Đã thêm ảnh: {os.path.basename(path)} ({i+1}/{len(image_paths)})")

out.release()
print(f"🎬 Video đã được tạo {output}")
