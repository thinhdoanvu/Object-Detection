import os
import numpy as np

# ==== CONFIG ====
INPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test"
SUB_DIR = "split_by_patient"
INPUT_PATH = os.path.join(INPUT_DIR, SUB_DIR)  # ✅ dùng thư mục chứa các file txt
OUTLIER_DIR = os.path.join(INPUT_DIR, "outlier")
REMOVED_DIR = os.path.join(INPUT_DIR, "removed_outlier")
# ================

os.makedirs(OUTLIER_DIR, exist_ok=True)
os.makedirs(REMOVED_DIR, exist_ok=True)


def detect_outlier(data):
    """Detect outlier using IQR rule."""
    if len(data) < 4:  # tránh lỗi percentile khi dữ liệu quá ít
        return np.array([False] * len(data))

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (data < lower) | (data > upper)


def process_file(file_path):
    """Đọc file .txt, phát hiện outlier và lưu kết quả."""
    lines, cx_list, cy_list = [], [], []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            lines.append(line.strip())
            _, _, _, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cx_list.append(cx)
            cy_list.append(cy)

    if not lines:
        print(f"[!] {os.path.basename(file_path)} không có dữ liệu hợp lệ.")
        return

    cx, cy = np.array(cx_list), np.array(cy_list)
    outlier_mask = detect_outlier(cx) | detect_outlier(cy)

    normal_lines = [l for i, l in enumerate(lines) if not outlier_mask[i]]
    outlier_lines = [l for i, l in enumerate(lines) if outlier_mask[i]]

    base_name = os.path.basename(file_path)
    out_path_removed = os.path.join(REMOVED_DIR, base_name)
    out_path_outlier = os.path.join(OUTLIER_DIR, base_name)

    with open(out_path_removed, "w") as f:
        f.write("\n".join(normal_lines))
    with open(out_path_outlier, "w") as f:
        f.write("\n".join(outlier_lines))

    print(f"✅ {base_name}: tổng {len(lines)} BB, outlier = {len(outlier_lines)}, còn lại = {len(normal_lines)}")


def main():
    txt_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".txt")]
    if not txt_files:
        print("❌ Không có file .txt nào trong thư mục đầu vào.")
        return

    for fname in txt_files:
        process_file(os.path.join(INPUT_PATH, fname))

    print("\n🎯 Hoàn tất! Các file đã được lưu:")
    print(f"- Outlier: {OUTLIER_DIR}")
    print(f"- Removed: {REMOVED_DIR}")


if __name__ == "__main__":
    main()
