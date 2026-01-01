import os

# ==== CONFIG ====
INPUT_FILE = r"C:\Users\VU\Documents\OBD\AICUP25\test\labels\t4_longest_best.txt"
OUTPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\labels"
IMAGE_WIDTH = 512   # <-- thay bằng width ảnh thật
IMAGE_HEIGHT = 512  # <-- thay bằng height ảnh thật
# ================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h

def split_lines(input_file, output_dir, img_w, img_h):
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            frame_id, cls, conf, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Bỏ confidence score, chỉ giữ cls + YOLO bbox
            x_center, y_center, w, h = convert_to_yolo(x1, y1, x2, y2, img_w, img_h)
            yolo_line = f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

            # Mỗi dòng -> 1 file riêng
            output_path = os.path.join(output_dir, f"{frame_id}.txt")
            with open(output_path, "w") as out_f:
                out_f.write(yolo_line + "\n")

            print(f"✅ Saved {output_path}")

    print("\nDone! Each line saved as separate YOLO file.")

if __name__ == "__main__":
    split_lines(INPUT_FILE, OUTPUT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
