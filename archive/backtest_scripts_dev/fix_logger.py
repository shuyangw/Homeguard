"""Quick script to fix logger calls."""

import re

with open('comprehensive_pairs_validation.py', 'r') as f:
    lines = f.readlines()

output_lines = []
for line in lines:
    # Remove all color= parameters
    line = re.sub(r',\s*color=["\'](?:green|red|yellow|blue)["\']', '', line)
    output_lines.append(line)

with open('comprehensive_pairs_validation.py', 'w') as f:
    f.writelines(output_lines)

print("Fixed all logger calls")
