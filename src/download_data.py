"""Download the CMU Keystroke Dynamics Benchmark into data/.

Usage (from the repository root):
    python src/download_data.py
"""
import os
import urllib.request

URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv")


def main():
    if os.path.exists(DEST):
        print(f"Already downloaded: {DEST}")
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    with open(DEST) as f:
        n_lines = sum(1 for _ in f)
    print(f"Saved {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB, {n_lines - 1} rows)")


if __name__ == "__main__":
    main()
