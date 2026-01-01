import cv2
import os
import matplotlib.pyplot as plt

LABEL_DIR = r"C:\Users\VU\Documents\OBD\datasets\Cardiomegaly\labels\val"
IMAGE_DIR = r"C:\Users\VU\Documents\OBD\datasets\Cardiomegaly\images\val"

def read_bbox_file(file_path):
    """Đọc file bbox, trả về list: [(class_id, cx, cy, w, h)]"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))  # xử lý cả số thập phân
            cx, cy, bw, bh = map(float, parts[1:5])
            bboxes.append((class_id, cx, cy, bw, bh))
    return bboxes

def visualize_images(label_dir, image_dir):
    # lọc thêm điều kiện dung lượng >0 KB
    label_files = [f for f in os.listdir(label_dir)
                   if f.endswith('.txt') and os.path.getsize(os.path.join(label_dir, f)) > 0]
    label_files.sort()
    if not label_files:
        print("Không có file label hợp lệ.")
        return

    fig, ax = plt.subplots()
    idx = 0
    print(f"Tổng số ảnh: {len(label_files)}. Nhấn Enter để qua ảnh kế tiếp, ESC để thoát.")

    while idx < len(label_files):
        label_name = label_files[idx]
        img_name = os.path.splitext(label_name)[0] + ".jpg"
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(img_path):
            print(f"[!] Không tìm thấy ảnh: {img_path}")
            idx += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Không thể đọc ảnh: {img_path}")
            idx += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Vẽ bbox
        h, w = img.shape[:2]
        bboxes = read_bbox_file(label_path)

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"{img_name} ({idx+1}/{len(label_files)})")
        ax.axis("off")

        for class_id, cx, cy, bw, bh in bboxes:
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            #ax.text(x1, max(y1-5,0), f"ID:{class_id}", color="yellow", fontsize=8, weight='bold')

        plt.pause(0.01)
        print(f"Đang hiển thị: {img_name} ({idx+1}/{len(label_files)})")

        key = input("Nhấn Enter để qua ảnh kế, ESC để thoát: ").strip().lower()
        if key == "esc" or key == "q":
            break
        idx += 1

    plt.close()

if __name__ == "__main__":
    visualize_images(LABEL_DIR, IMAGE_DIR)
