# -*- coding: utf-8 -*-
from collections import defaultdict

input_file = "../AICUP25/test/682.0.001_longest.txt"
output_file = "../AICUP25/test/682.0.001_longest_best.txt"

# Lưu dòng tốt nhất theo filename
best_lines = {}

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # bỏ dòng không đủ cột
        filename, cls, conf = parts[0], parts[1], float(parts[2])

        # Nếu filename chưa có hoặc confidence cao hơn thì cập nhật
        if filename not in best_lines or conf > best_lines[filename][2]:
            best_lines[filename] = (filename, cls, conf, *parts[3:])

# Ghi kết quả ra file
with open(output_file, "w") as f:
    for k in sorted(best_lines.keys()):
        f.write(" ".join(map(str, best_lines[k])) + "\n")

print(f"Filtered data saved to {output_file}")
