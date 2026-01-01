import os
import numpy as np

# ==== CONFIG ====
INPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test"
SUB_DIR = "split_by_patient"
INPUT_PATH = os.path.join(INPUT_DIR, SUB_DIR)
OUTLIER_DIR = os.path.join(INPUT_DIR, "outlier")
REMOVED_DIR = os.path.join(INPUT_DIR, "removed_outlier")
# =================

os.makedirs(OUTLIER_DIR, exist_ok=True)
os.makedirs(REMOVED_DIR, exist_ok=True)


def process_file(file_path):
    """Xử lý 1 file txt: lọc BB quá lớn dựa trên tỷ lệ >1.5"""
    lines, bboxes = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            lines.append(line.strip())
            _, _, _, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            bboxes.append((x1, y1, x2, y2))

    if not lines:
        print(f"[!] {os.path.basename(file_path)} không có dữ liệu hợp lệ.")
        return

    bboxes = np.array(bboxes)
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    ratios = np.maximum(widths / heights, heights / widths)
    outlier_mask = ratios > 1.25

    normal_lines = [l for i, l in enumerate(lines) if not outlier_mask[i]]
    outlier_lines = [l for i, l in enumerate(lines) if outlier_mask[i]]

    # Lưu file
    base_name = os.path.basename(file_path)
    with open(os.path.join(REMOVED_DIR, base_name), "w", encoding="utf-8") as f:
        f.write("\n".join(normal_lines))
    with open(os.path.join(OUTLIER_DIR, base_name), "w", encoding="utf-8") as f:
        f.write("\n".join(outlier_lines))

    print(f"✅ {base_name}: tổng {len(lines)} BB, outlier = {len(outlier_lines)}, còn lại = {len(normal_lines)}")


def main():
    txt_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".txt")]
    if not txt_files:
        print("Không có file .txt nào trong thư mục split_by_patient.")
        return

    for fname in txt_files:
        process_file(os.path.join(INPUT_PATH, fname))

    print("\n🎯 Hoàn tất! Các file đã được lưu:")
    print(f"- Outlier: {OUTLIER_DIR}")
    print(f"- Removed: {REMOVED_DIR}")


if __name__ == "__main__":
    main()
