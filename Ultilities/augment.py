"""DICH ANHR L< R< D< U"""
import os
import cv2
import random
import numpy as np
from pathlib import Path

class CropTranslate:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.lbl_dir = self.root_dir / "labels"
        self.out_img = self.root_dir / "aug_img"
        self.out_lbl = self.root_dir / "aug_lbl"
        self.out_img.mkdir(parents=True, exist_ok=True)
        self.out_lbl.mkdir(parents=True, exist_ok=True)

    def xywhn2xyxy(self, bboxes, w, h):
        out = []
        for xc, yc, bw, bh in bboxes:
            x_min = (xc - bw/2) * w
            y_min = (yc - bh/2) * h
            x_max = (xc + bw/2) * w
            y_max = (yc + bh/2) * h
            out.append([x_min, y_min, x_max, y_max])
        return np.array(out, dtype=np.float32)

    def xyxy2xywhn(self, bboxes, w, h):
        out = []
        for x_min, y_min, x_max, y_max in bboxes:
            xc = ((x_min + x_max) / 2) / w
            yc = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            out.append([xc, yc, bw, bh])
        return np.array(out, dtype=np.float32)

    def translate_image(self, img, bboxes_px, side, shift_px):
        h, w = img.shape[:2]
        translated = np.zeros_like(img)

        if side == "r":  # dịch sang phải
            translated[:, shift_px:] = img[:, :w-shift_px]
            offset_x, offset_y = shift_px, 0
        elif side == "l":  # dịch sang trái
            translated[:, :w-shift_px] = img[:, shift_px:]
            offset_x, offset_y = -shift_px, 0
        elif side == "t":  # dịch lên trên
            translated[:h-shift_px, :] = img[shift_px:, :]
            offset_x, offset_y = 0, -shift_px
        else:  # dịch xuống dưới
            translated[shift_px:, :] = img[:h-shift_px, :]
            offset_x, offset_y = 0, shift_px

        # dịch box theo offset
        new_boxes = []
        for x_min, y_min, x_max, y_max in bboxes_px:
            x_min += offset_x
            x_max += offset_x
            y_min += offset_y
            y_max += offset_y
            # clip
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            if x_max > x_min and y_max > y_min:
                new_boxes.append([x_min, y_min, x_max, y_max])

        return translated, np.array(new_boxes, dtype=np.float32)

    def process(self):
        for lbl_file in self.lbl_dir.glob("*.txt"):
            lines = [l.strip().split() for l in open(lbl_file).readlines()]
            if len(lines) == 0:
                continue  # skip empty labels

            # đọc ảnh tương ứng
            img_file = self.img_dir / (lbl_file.stem + ".png")
            if not img_file.exists():
                img_file = self.img_dir / (lbl_file.stem + ".jpg")
            if not img_file.exists():
                print(f"[Skip] Image not found for {lbl_file}")
                continue

            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]

            cls = np.array([[int(x[0])] for x in lines], dtype=np.float32)
            bboxes = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in lines], dtype=np.float32)

            # convert sang pixel xyxy
            bboxes_px = self.xywhn2xyxy(bboxes, w, h)

            for side in ["l", "r", "t", "b"]:
                shift_px = random.randint(10, 50)
                translated, new_boxes = self.translate_image(img, bboxes_px, side, shift_px)

                if len(new_boxes) == 0:
                    continue

                # convert lại sang normalized xywh theo kích thước gốc (w,h)
                bboxes_n = self.xyxy2xywhn(new_boxes, w, h)

                # lưu ảnh
                out_img_file = self.out_img / f"{lbl_file.stem}_{side}.png"
                cv2.imwrite(str(out_img_file), translated)

                # lưu label
                out_lbl_file = self.out_lbl / f"{lbl_file.stem}_{side}.txt"
                with open(out_lbl_file, "w") as f:
                    for i, bb in enumerate(bboxes_n):
                        f.write(f"{int(cls[i][0])} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

                print(f"[Saved] {out_img_file}, {out_lbl_file}")


if __name__ == "__main__":
    root = r"C:\Users\VU\Documents\OBD\AICUP25\gin_train"
    cropper = CropTranslate(root_dir=root)
    cropper.process()
