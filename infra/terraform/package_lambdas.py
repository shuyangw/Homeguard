"""
Package Lambda functions for AWS deployment.

Creates ZIP files with proper structure for AWS Lambda.
"""
import zipfile
from pathlib import Path

def package_lambda(source_file: str, output_zip: str):
    """Package a Lambda function as index.py in a ZIP file."""
    source_path = Path(source_file)
    output_path = Path(output_zip)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the Python file as index.py (Lambda expects this name)
        zipf.write(source_path, arcname='index.py')

    print(f"[OK] Created {output_path.name} ({output_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    # Package start instance Lambda
    package_lambda(
        'lambda/start_instance.py',
        'lambda_start_instance.zip'
    )

    # Package stop instance Lambda
    package_lambda(
        'lambda/stop_instance.py',
        'lambda_stop_instance.zip'
    )

    print("\n[OK] All Lambda functions packaged successfully!")
    print("  - lambda_start_instance.zip")
    print("  - lambda_stop_instance.zip")
