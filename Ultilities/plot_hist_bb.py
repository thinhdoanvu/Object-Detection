import numpy as np
import matplotlib.pyplot as plt

# --- Cấu hình ---
TEST_IMG_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\images"  # Thư mục test
RESULT_FILE = r"C:\Users\VU\Documents\OBD\AICUP25\test\t10v255.txt"
IMG_WIDTH = 512   # thay bằng kích thước thật của ảnh test
IMG_HEIGHT = 512  # thay bằng kích thước thật của ảnh test

# --- Khởi tạo ma trận heatmap ---
heatmap = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

# --- Đọc file kết quả inference ---
with open(RESULT_FILE, "r") as f:
    lines = f.read().strip().splitlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 7:
        continue
    _, _, conf, x1, y1, x2, y2 = parts
    conf = float(conf)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Clip bounding box để không vượt ngoài ảnh
    x1 = max(0, min(x1, IMG_WIDTH-1))
    x2 = max(0, min(x2, IMG_WIDTH-1))
    y1 = max(0, min(y1, IMG_HEIGHT-1))
    y2 = max(0, min(y2, IMG_HEIGHT-1))

    # Cộng confidence vào heatmap
    heatmap[y1:y2+1, x1:x2+1] += conf

# --- Chuẩn hóa heatmap để hiển thị ---
heatmap = np.clip(heatmap, 0, np.percentile(heatmap, 99))  # tránh outlier quá sáng

# --- Vẽ heatmap ---
plt.figure(figsize=(8, 8))
plt.imshow(heatmap, cmap='hot', origin='upper')
plt.colorbar(label='Accumulated confidence')
plt.title('Bounding Box Density Heatmap')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.show()
