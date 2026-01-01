import os

# Thư mục chứa các file txt cần nối
INPUT_DIR = r"C:\Users\VU\Documents\OBD\AICUP25\test\square\outlier"
# Đường dẫn file output
OUTPUT_FILE = os.path.join(INPUT_DIR, "outlier.txt")

# Mở file kết quả để ghi
with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    # Duyệt qua tất cả file .txt trong thư mục
    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.endswith(".txt"):
            fpath = os.path.join(INPUT_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as infile:
                # Ghi nội dung của file hiện tại vào file tổng
                outfile.write(infile.read())
                # Thêm dòng trống giữa các file (nếu muốn)
                outfile.write("\n")

print(f"✅ Đã nối xong tất cả file .txt thành: {OUTPUT_FILE}")
