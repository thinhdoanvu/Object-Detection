from collections import defaultdict

file_path = r"C:\Users\VU\Documents\OBD\AICUP25\test\756.txt"
output_path = r"C:\Users\VU\Documents\OBD\AICUP25\test\756_longest.txt"

# --- Đọc file ---
lines = [line.strip() for line in open(file_path, "r") if line.strip()]

# --- Gom các dòng theo patient ---
patients_lines = defaultdict(list)
for line in lines:
    patient_full = line.split()[0]  # ví dụ patient0051_0224
    patient_id = '_'.join(patient_full.split('_')[:-1])  # patient0051
    idx = int(patient_full.split('_')[-1])  # 0224
    patients_lines[patient_id].append((idx, line))

# --- Tìm dãy liên tiếp dài nhất mỗi patient ---
longest_sequences = []

for patient_id, entries in patients_lines.items():
    # sắp xếp theo số thứ tự
    entries.sort(key=lambda x: x[0])

    current_seq = []
    longest_seq = []
    last_idx = None

    for idx, line in entries:
        if last_idx is None or idx == last_idx or idx == last_idx + 1:
            # nếu cùng số hoặc liên tiếp
            current_seq.append((idx, line))
        else:
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq.copy()
            current_seq = [(idx, line)]
        last_idx = idx

    # kiểm tra cuối patient
    if len(current_seq) > len(longest_seq):
        longest_seq = current_seq.copy()

    # thêm vào kết quả
    longest_sequences.extend([line for _, line in longest_seq])

# --- Ghi ra file ---
with open(output_path, "w") as f:
    for line in longest_sequences:
        f.write(line + "\n")

print(f"Hoàn tất! Đã lưu dãy liên tiếp dài nhất mỗi patient vào {output_path}")
