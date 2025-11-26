"""
Script to standardize sys.path setup across all backtest scripts.

This script automatically updates all Python files in backtest_scripts/
to use the standardized path_setup module instead of custom sys.path
manipulation.

Replaces patterns like:
    - sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    - sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    - sys.path.insert(0, str(Path(__file__).parent.parent))
    - sys.path.append(str(Path(__file__).parent.parent / 'src'))

With:
    from utils.path_setup import setup_project_paths
    ROOT_DIR = setup_project_paths()

Usage:
    python backtest_scripts/utils/standardize_paths.py [--dry-run]
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_sys_path_lines(content: str) -> List[Tuple[int, str]]:
    """
    Find all sys.path manipulation lines in the content.

    Args:
        content: The file content as a string

    Returns:
        List of (line_number, line_content) tuples
    """
    lines = content.split('\n')
    sys_path_lines = []

    for i, line in enumerate(lines):
        # Match various sys.path patterns
        if re.search(r'sys\.path\.(insert|append|extend)', line):
            sys_path_lines.append((i, line))

    return sys_path_lines


def has_path_setup_import(content: str) -> bool:
    """Check if file already imports path_setup."""
    return 'from utils.path_setup import' in content or 'import utils.path_setup' in content


def standardize_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Standardize sys.path setup in a single file.

    Args:
        file_path: Path to the Python file
        dry_run: If True, only report changes without modifying file

    Returns:
        Tuple of (was_modified, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already using standardized path setup
        if has_path_setup_import(content):
            return False, "Already using standardized path setup"

        # Find sys.path lines
        sys_path_lines = find_sys_path_lines(content)
        if not sys_path_lines:
            return False, "No sys.path manipulation found"

        # Split content into lines for modification
        lines = content.split('\n')

        # Find import section (after docstring, before first non-import code)
        import_section_end = 0
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    docstring_char = stripped[:3]
                    in_docstring = True
                    if stripped.count(docstring_char) >= 2:  # Single-line docstring
                        in_docstring = False
                elif docstring_char in stripped:
                    in_docstring = False
                continue

            # Skip blank lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            # Found import or from statement
            if in_docstring:
                continue

            if stripped.startswith('import ') or stripped.startswith('from '):
                import_section_end = i + 1
            elif import_section_end > 0:
                # First non-import line after imports
                break

        # Remove old sys.path lines
        for line_num, _ in reversed(sys_path_lines):
            del lines[line_num]

        # Find where to insert new import
        # Look for existing Path import from pathlib
        path_import_idx = None
        sys_import_idx = None
        last_stdlib_import_idx = 0

        for i, line in enumerate(lines[:import_section_end + 5]):  # Check near import section
            stripped = line.strip()
            if 'from pathlib import Path' in stripped:
                path_import_idx = i
            if 'import sys' in stripped and not stripped.startswith('#'):
                sys_import_idx = i
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Track stdlib imports
                if not any(lib in stripped for lib in ['pandas', 'numpy', 'vectorbt', 'matplotlib', 'src.', 'strategies.', 'engine.', 'utils.']):
                    last_stdlib_import_idx = i

        # Insert new standardized import after stdlib imports
        insert_idx = last_stdlib_import_idx + 1
        new_import = "from utils.path_setup import setup_project_paths"
        new_setup = "ROOT_DIR = setup_project_paths()"

        # Add blank line before our import if needed
        if insert_idx > 0 and lines[insert_idx - 1].strip():
            lines.insert(insert_idx, "")
            insert_idx += 1

        lines.insert(insert_idx, new_import)
        lines.insert(insert_idx + 1, new_setup)

        # Join back into content
        new_content = '\n'.join(lines)

        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

        return True, f"Updated {len(sys_path_lines)} sys.path line(s)"

    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Standardize sys.path setup in backtest scripts")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    args = parser.parse_args()

    # Find all Python files in backtest_scripts
    backtest_dir = Path(__file__).parent.parent
    python_files = list(backtest_dir.glob('*.py'))

    print(f"Found {len(python_files)} Python files in backtest_scripts/")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    print("-" * 80)

    modified_count = 0
    skipped_count = 0
    error_count = 0

    for py_file in sorted(python_files):
        was_modified, message = standardize_file(py_file, dry_run=args.dry_run)

        status = "[MODIFIED]" if was_modified else "[SKIPPED] "
        print(f"{status} {py_file.name:50} - {message}")

        if was_modified:
            modified_count += 1
        elif "Error" in message:
            error_count += 1
        else:
            skipped_count += 1

    print("-" * 80)
    print(f"Summary:")
    print(f"  Modified: {modified_count}")
    print(f"  Skipped:  {skipped_count}")
    print(f"  Errors:   {error_count}")

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
