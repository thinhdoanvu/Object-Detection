import cv2
import os

LABEL_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\42_training_label\patient0001"
IMAGE_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\42_training_image\patient0001"
OUTPUT_VIDEO = r"C:\Users\VU\Documents\OBD\AICUP25\train\gabor.mp4"
DISPLAY_TIME_MS = 200  # thời gian hiển thị mỗi ảnh

def read_bbox_file(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            bboxes.append((class_id, cx, cy, w, h))
    return bboxes

def show_and_save(label_dir, image_dir, output_path=None, display_time_ms=200):
    img_files = [f for f in os.listdir(image_dir) if f.endswith(('.png','.jpg','.jpeg'))]
    img_files.sort()
    if not img_files:
        print("Không có ảnh hợp lệ.")
        return

    # chuẩn bị VideoWriter nếu cần
    video_writer = None
    if output_path:
        first_img = cv2.imread(os.path.join(image_dir, img_files[0]))
        h, w = first_img.shape[:2]
        fps = 1000 / display_time_ms
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_name in img_files:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        bboxes = read_bbox_file(label_path)

        for class_id, cx, cy, bw, bh in bboxes:
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, str(class_id), (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # hiển thị trực tiếp
        cv2.imshow("Patient Images", img)
        key = cv2.waitKey(display_time_ms) & 0xFF
        if key == 27:  # ESC để thoát
            break

        # ghi video nếu có
        if video_writer:
            video_writer.write(img)

    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
        print(f"✅ Video đã lưu tại: {output_path}")

if __name__ == "__main__":
    # Nếu muốn tạo video thì truyền OUTPUT_VIDEO
    #show_and_save(LABEL_DIR, IMAGE_DIR, OUTPUT_VIDEO, DISPLAY_TIME_MS)

    # Nếu không cần tạo video thì comment dòng trên và dùng dòng dưới:
    show_and_save(LABEL_DIR, IMAGE_DIR, None, DISPLAY_TIME_MS)
