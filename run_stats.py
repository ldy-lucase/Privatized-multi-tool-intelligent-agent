#!/usr/bin/env python3
"""统计 /home/yun/deepagents/ 下所有 .py 文件的行数（排除 .venv）。"""

import os
import glob

ROOT = '/home/yun/deepagents'
OUTPUT = os.path.join(ROOT, 'stats.txt')

# Find all .py files, excluding .venv
all_py = glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True)
py_files = [f for f in all_py if '.venv' not in f]

total_lines = 0
details = []

for f in sorted(py_files):
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = sum(1 for _ in fh)
            total_lines += lines
            details.append((f, lines))
    except Exception as e:
        details.append((f, f'error: {e}'))

with open(OUTPUT, 'w') as out:
    out.write('Python File Line Statistics\n')
    out.write('=' * 60 + '\n\n')
    for path, count in details:
        out.write(f'{path}: {count} lines\n')
    out.write('\n' + '=' * 60 + '\n')
    out.write(f'Total .py files: {len(py_files)}\n')
    out.write(f'Total lines: {total_lines}\n')

print(f'Done. {len(py_files)} files, {total_lines} total lines.')
print(f'Result saved to {OUTPUT}')
