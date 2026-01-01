"""
Lon hon 1.5 lan BB la MAX accuracy thi delete
"""

import os

DATA_FILE = r"C:\Users\VU\Documents\OBD\AICUP25\test\9m001_remove9.txt"
OUTPUT_FILE = r"C:\Users\VU\Documents\OBD\AICUP25\test\9m001_filtered.txt"

def read_bbox_file(file_path):
    """
    Đọc file bbox, trả về dict: {image_name: [(score, x1, y1, x2, y2, line_text), ...]}
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            name = parts[0]
            score = float(parts[2])
            x1, y1, x2, y2 = map(int, parts[3:7])
            data.setdefault(name, []).append((score, x1, y1, x2, y2, line.strip()))
    return data

def filter_large_bbox(data, scale=1.5):
    """
    Loại bỏ BB có diện tích > scale * diện tích BB có score lớn nhất
    """
    filtered_data = {}
    for name, bboxes in data.items():
        # Tìm diện tích của BB có score cao nhất
        max_score = max(b[0] for b in bboxes)
        max_area = 0
        for b in bboxes:
            if b[0] == max_score:
                _, x1, y1, x2, y2, _ = b
                max_area = (x2 - x1) * (y2 - y1)
                break

        # Lọc các BB có diện tích <= 1.5 * max_area
        filtered = []
        for b in bboxes:
            score, x1, y1, x2, y2, line_text = b
            area = (x2 - x1) * (y2 - y1)
            if area <= scale * max_area:
                filtered.append(b)
        if filtered:
            filtered_data[name] = filtered
    return filtered_data

def save_filtered_data(filtered_data, output_file):
    with open(output_file, 'w') as f:
        for bboxes in filtered_data.values():
            for b in bboxes:
                f.write(b[5] + '\n')

if __name__ == "__main__":
    data = read_bbox_file(DATA_FILE)
    filtered = filter_large_bbox(data, scale=1.5)
    save_filtered_data(filtered, OUTPUT_FILE)
    print(f"Đã lọc xong. Kết quả lưu vào: {OUTPUT_FILE}")
