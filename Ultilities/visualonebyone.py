import cv2
import os
import matplotlib.pyplot as plt

# ==== CONFIG ====
DATA_FILE = r"C:\Users\VU\Documents\OBD\AICUP25\test\res\t4_longest.txt"
IMAGE_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\images"
# =================

def read_bbox_file(file_path):
    """Đọc file bbox, trả về dict: {image_name: [(score, x1, y1, x2, y2), ...]}"""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            name = parts[0]
            score = float(parts[2])
            x1, y1, x2, y2 = map(int, parts[3:7])
            data.setdefault(name, []).append((score, x1, y1, x2, y2))
    return data

def visualize_images(data):
    image_names = list(data.keys())
    idx = 0

    print(f"Tổng số ảnh: {len(image_names)}. Nhấn → hoặc Enter để qua ảnh kế tiếp, ESC để thoát.")

    fig, ax = plt.subplots()
    while True:
        name = image_names[idx]
        img_path = os.path.join(IMAGE_DIR, name + ".png")
        if not os.path.exists(img_path):
            print(f"[!] Không tìm thấy ảnh: {img_path}")
            idx = (idx + 1) % len(image_names)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Không thể đọc ảnh: {img_path}")
            idx = (idx + 1) % len(image_names)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"{name} ({idx+1}/{len(image_names)})")
        ax.axis("off")

        # Vẽ bounding boxes
        for (score, x1, y1, x2, y2) in data[name]:
            color = "lime" if score > 0.1 else "red"
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score:.4f}", color=color, fontsize=8, weight='bold')

        plt.pause(0.01)

        print(f"Đang hiển thị: {name} ({idx+1}/{len(image_names)})")
        key = input("Nhấn Enter hoặc → để qua ảnh kế, ESC để thoát: ").strip().lower()
        if key == "esc" or key == "q":
            break
        idx = (idx + 1) % len(image_names)

    plt.close()

if __name__ == "__main__":
    bbox_data = read_bbox_file(DATA_FILE)
    print(f"Đã đọc {len(bbox_data)} ảnh hợp lệ từ {DATA_FILE}")
    visualize_images(bbox_data)
