import sys
from pathlib import Path

file_path = Path(__file__).parent / 'test_trading_engine_e2e.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters with ASCII
content = content.replace('✓', '[OK]')
content = content.replace('✗', '[FAIL]')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Replaced all Unicode characters in {file_path.name}')
