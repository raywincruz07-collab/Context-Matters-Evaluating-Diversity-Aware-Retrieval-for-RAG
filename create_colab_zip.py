"""
Creates medical-rag-maki-colab-sprint2.zip for Colab deployment.

Files are stored with paths relative to the project root (e.g. src/config.py),
so the zip extracts flat on Colab without an extra nesting folder.
"""

import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(PROJECT_ROOT, "medical-rag-maki-colab-sprint2.zip")

# Directories to walk recursively (relative to PROJECT_ROOT).
# demo/ is intentionally excluded — it holds the optional Streamlit UI
# (app.py + pubmed_fetch.py) which is not part of the evaluated pipeline.
INCLUDE_DIRS = ["src", "tests"]

# Individual root-level files to include
INCLUDE_FILES = ["requirements.txt", "README.md"]

# Directory names to skip entirely while walking
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ragatouille", ".git"}


def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS


def collect_dir_entries(rel_dir: str):
    """Yield (abs_path, zip_arcname) for every non-skipped file under rel_dir."""
    abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
    for dirpath, dirnames, filenames in os.walk(abs_dir):
        # Prune skip-dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            if fname.endswith(".pyc"):
                continue
            abs_file = os.path.join(dirpath, fname)
            # arcname: path relative to PROJECT_ROOT, using forward slashes
            arcname = os.path.relpath(abs_file, PROJECT_ROOT).replace(os.sep, "/")
            yield abs_file, arcname


included = []  # list of arcnames added

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # Walk src/ and tests/
    for rel_dir in INCLUDE_DIRS:
        for abs_file, arcname in collect_dir_entries(rel_dir):
            zf.write(abs_file, arcname)
            included.append(arcname)

    # Root-level files
    for fname in INCLUDE_FILES:
        abs_file = os.path.join(PROJECT_ROOT, fname)
        if os.path.exists(abs_file):
            zf.write(abs_file, fname)
            included.append(fname)
        else:
            print(f"WARNING: {fname} not found, skipping.")

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)

print(f"\nZip created : {ZIP_PATH}")
print(f"Files included : {len(included)}")
print(f"Zip size       : {zip_size_mb:.2f} MB")
print("\nIncluded files:")
for f in sorted(included):
    print(f"  {f}")
