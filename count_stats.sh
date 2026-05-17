#!/bin/bash
# 统计 /home/yun/deepagents/ 下所有 .py 文件的行数（排除 .venv）

ROOT="/home/yun/deepagents"
OUTPUT="$ROOT/stats.txt"

echo "Python File Line Statistics" > "$OUTPUT"
echo "============================================================" >> "$OUTPUT"
echo "" >> "$OUTPUT"

TOTAL_LINES=0
FILE_COUNT=0

# 查找所有 .py 文件，排除 .venv
while IFS= read -r file; do
    if [[ "$file" == *".venv"* ]]; then
        continue
    fi
    
    # 统计行数
    lines=$(wc -l < "$file")
    TOTAL_LINES=$((TOTAL_LINES + lines))
    FILE_COUNT=$((FILE_COUNT + 1))
    
    echo "$file: $lines lines" >> "$OUTPUT"
done < <(find "$ROOT" -name "*.py" -type f)

echo "" >> "$OUTPUT"
echo "============================================================" >> "$OUTPUT"
echo "Total .py files: $FILE_COUNT" >> "$OUTPUT"
echo "Total lines: $TOTAL_LINES" >> "$OUTPUT"

echo "Done. $FILE_COUNT files, $TOTAL_LINES total lines."
echo "Result saved to $OUTPUT"
