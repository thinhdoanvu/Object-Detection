import numpy as np
from collections import defaultdict

def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def filter_boxes(file_path, save_path="filtered_boxes.txt", edge_tol=5):
    with open(file_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]
    frames = defaultdict(list)
    for line in lines:
        frame_id, cls, score, x1, y1, x2, y2 = line
        score = float(score)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        area = box_area(x1, y1, x2, y2)
        frames[frame_id].append((frame_id, cls, score, x1, y1, x2, y2, area))

    final = []

    for frame_id, boxes in frames.items():
        n = len(boxes)

        if n == 1:
            final.extend(boxes)

        elif n == 2:
            b1, b2 = boxes
            iou_val = iou(b1[3:7], b2[3:7])
            if iou_val == 0:
                best = max(boxes, key=lambda b: b[2])  # score cao hơn
                final.append(best)
            else:
                # overlap → xét cạnh
                if abs(b1[5] - b2[5]) <= edge_tol:  # cạnh phải gần nhau (cho phép bằng 0)
                    best = min(boxes, key=lambda b: b[7])  # diện tích nhỏ nhất
                    final.append(best)
                elif abs(b1[3] - b2[3]) <= edge_tol:  # cạnh trái gần nhau (cho phép bằng 0)
                    best = max(boxes, key=lambda b: b[7])  # diện tích lớn nhất
                    final.append(best)
                else:
                    final.extend(boxes)

        else:  # n >= 3
            # loại bỏ box không overlap
            overlapping = []
            for i, bi in enumerate(boxes):
                ov = any(iou(bi[3:7], bj[3:7]) > 0 for j, bj in enumerate(boxes) if j != i)
                if ov:
                    overlapping.append(bi)

            if len(overlapping) == 0:
                continue

            kept = []
            removed = set()

            # cạnh trái gần nhau → giữ diện tích lớn nhất
            overlapping_sorted_x1 = sorted(range(len(overlapping)), key=lambda k: overlapping[k][3])
            for a, b in zip(overlapping_sorted_x1, overlapping_sorted_x1[1:]):
                if a in removed or b in removed: continue
                if abs(overlapping[a][3] - overlapping[b][3]) <= edge_tol:
                    keep_idx = a if overlapping[a][7] >= overlapping[b][7] else b
                    drop_idx = b if keep_idx == a else a
                    removed.add(drop_idx)

            # cạnh phải gần nhau → giữ diện tích nhỏ nhất
            overlapping_sorted_x2 = sorted(range(len(overlapping)), key=lambda k: overlapping[k][5])
            for a, b in zip(overlapping_sorted_x2, overlapping_sorted_x2[1:]):
                if a in removed or b in removed: continue
                if abs(overlapping[a][5] - overlapping[b][5]) <= edge_tol:
                    keep_idx = a if overlapping[a][7] <= overlapping[b][7] else b
                    drop_idx = b if keep_idx == a else a
                    removed.add(drop_idx)

            for i, b in enumerate(overlapping):
                if i not in removed:
                    kept.append(b)

            final.extend(kept)

    # Lưu ra file
    with open(save_path, "w") as f:
        for frame_id, cls, score, x1, y1, x2, y2, area in final:
            f.write(f"{frame_id} {cls} {score:.4f} {x1} {y1} {x2} {y2}\n")

    print(f"[Done] Saved filtered results to {save_path}")


# Ví dụ chạy
file_path = r"C:\\Users\\VU\\Documents\\OBD\\AICUP25\\test\\res\\t4.0.001.txt"
save_path = r"C:\\Users\\VU\\Documents\\OBD\\AICUP25\\test\\t4v6.txt"
filter_boxes(file_path, save_path, edge_tol=2)
