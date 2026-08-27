# ============ Step 1 验收：src/check_data.py（独立文件，专门数数） ============
import glob
from collections import Counter

# glob.glob 列出所有符合条件的文件；** 表示任意子目录，recursive=True 打开递归
files = glob.glob("data/**/*.wav", recursive=True)
print("wav 文件总数：", len(files))

# Counter 就是个智能计数器：每见一个键，该键计数 +1
emotion_codes = Counter()
for f in files:
    code = f.split("-")[2]      # 文件名按 - 切开取第 3 段 = 情感码
    emotion_codes[code] += 1

for code in sorted(emotion_codes):
    print(f"情感 {code}: {emotion_codes[code]} 条")