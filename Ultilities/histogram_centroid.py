import matplotlib.pyplot as plt
import numpy as np

txt_path = r"C:\Users\VU\Documents\OBD\AICUP25\test\split_by_patient\patient0055.txt"

cx_list, cy_list = [], []

# --- đọc file và tính tọa độ trung tâm ---
with open(txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        _, _, _, x1, y1, x2, y2 = parts
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cx_list.append(cx)
        cy_list.append(cy)

cx = np.array(cx_list)
cy = np.array(cy_list)

print(f"Tổng số bounding boxes: {len(cx)}")

# --- Xác định outlier bằng IQR ---
def detect_outlier(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (data < lower) | (data > upper)

outlier_x = detect_outlier(cx)
outlier_y = detect_outlier(cy)
outlier_mask = outlier_x | outlier_y

# --- Vẽ ---
plt.figure(figsize=(8, 6))
plt.scatter(cx[~outlier_mask], cy[~outlier_mask], s=8, c='dodgerblue', alpha=0.6, label="Normal")
plt.scatter(cx[outlier_mask], cy[outlier_mask], s=12, c='red', alpha=0.9, label="Outlier")

plt.title("Distribution of Bounding Box Centers (with Outliers)")
plt.xlabel("Center X")
plt.ylabel("Center Y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Số lượng outlier: {outlier_mask.sum()}")
