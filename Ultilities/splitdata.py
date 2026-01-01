import os
import random
import shutil
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    # Lấy danh sách file ảnh (đuôi .jpg, .png, .jpeg)
    exts = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]
    random.shuffle(images)

    n_val = int(len(images) * val_ratio)
    val_images = images[:n_val]
    train_images = images[n_val:]

    # Tạo thư mục output
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Hàm copy ảnh + label
    def copy_files(file_list, split):
        for img_file in tqdm(file_list, desc=f"Copying {split}", unit="file"):
            # Copy ảnh
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, "images", split, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy label nếu có
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, "labels", split, label_file)
                shutil.copy2(src_label, dst_label)

    # Chia train/val
    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print(f"✅ Done! {len(train_images)} train images, {len(val_images)} val images.")

if __name__ == "__main__":
    images_dir = "../datasets/aortic_valve/images"   # Thư mục ảnh gốc
    labels_dir = "../datasets/aortic_valve/labels"   # Thư mục nhãn gốc
    output_dir = "../datasets/aortic_valve/dataset_yolo"     # Thư mục output sau khi chia
    split_dataset(images_dir, labels_dir, output_dir, val_ratio=0.2)
