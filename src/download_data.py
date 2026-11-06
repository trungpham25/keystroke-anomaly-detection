"""Download the CMU Keystroke Dynamics Benchmark into data/.

Usage (from the repository root):
    python src/download_data.py

Other scripts (load_data.py, run_all.py) call ensure_data(), which downloads
the file only if it is not already in data/.

To use a different mirror (or a local copy for testing), set the environment
variable KEYSTROKE_URL, e.g.  KEYSTROKE_URL=file:///path/to/file.csv
"""
import os
import urllib.request

URL = os.environ.get(
    "KEYSTROKE_URL", "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv")

EXPECTED_HEADER_START = "subject,sessionIndex,rep,H.period"
EXPECTED_ROWS = 20400


def _validate(path):
    """Return the number of data rows, or raise if this is not the right file."""
    with open(path, newline="") as f:
        header = f.readline().strip()
        n_rows = sum(1 for _ in f)
    if not header.startswith(EXPECTED_HEADER_START):
        raise ValueError(
            f"Downloaded file does not look like the keystroke CSV "
            f"(unexpected header: {header[:60]!r}). The server may have returned "
            f"an error page."
        )
    if n_rows != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, found {n_rows}.")
    return n_rows


def ensure_data(verbose=True):
    """Make sure the CSV exists in data/, downloading it if needed.

    Returns the path to the CSV. Raises RuntimeError with manual-download
    instructions if the download fails.
    """
    if os.path.exists(DEST):
        if verbose:
            print(f"Already downloaded: {DEST}")
        return DEST

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    tmp = DEST + ".part"  # download to a temp name so a failure leaves no bad file
    try:
        if verbose:
            print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, tmp)
        n_rows = _validate(tmp)
        os.replace(tmp, DEST)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Could not download the dataset ({type(e).__name__}: {e}).\n"
            f"Download it manually from {URL}\n"
            f"and save it as {DEST}"
        ) from e

    if verbose:
        print(f"Saved {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB, {n_rows} rows)")
    return DEST


def main():
    ensure_data()


if __name__ == "__main__":
    main()
