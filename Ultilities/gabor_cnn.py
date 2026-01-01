import cv2
import numpy as np
import os
from tqdm import tqdm

# ==== CONFIG ====
INPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\images"
OUTPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\enhanced_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================

def linear_contrast_stretch(img):
    """Kéo dải cường độ sáng về toàn 0–255."""
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val < 1:
        return img
    stretched = (img - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)

def enhance_smooth_dark_edges(img):
    # 1️⃣ Blur mạnh vùng sáng
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    # 2️⃣ Linear contrast để đen nổi bật
    enhanced = linear_contrast_stretch(blurred)

    # 3️⃣ Optional: tăng chút độ tương phản để làm đường đen rõ hơn
    alpha = 1.5  # contrast
    beta = -30   # làm nền sáng tối đi
    final = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    return final

def process_all_images(input_dir, output_dir):
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in tqdm(images, desc="Enhancing contrast"):
        in_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)

        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[!] Lỗi đọc ảnh: {img_name}")
            continue

        enhanced = enhance_smooth_dark_edges(img)
        cv2.imwrite(out_path, enhanced)

    print(f"✅ Hoàn tất! Ảnh tăng cường đã lưu tại: {output_dir}")

if __name__ == "__main__":
    process_all_images(INPUT_DIR, OUTPUT_DIR)
