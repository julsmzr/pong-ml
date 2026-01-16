import gdown
import zipfile
import os

VERBOSE = False


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)

def download(url: str, zip_path: str, verbose: bool) -> None:
    """Downloads file from Google Drive URL to specified path."""
    gdown.download(url, zip_path, quiet=False)

    if not os.path.exists(zip_path):
        print("The dataset could not be downloaded sucessfully.")
        print(f"Please download it manully using the url below and place it in the root directory and rename to {zip_path}.\n")

        print("-" * 50)
        print(url)
        print("-" * 50)

        input("\nPress Enter to continue or Ctrl+C to exit.")

def extract(zip_path: str, extract_dir: str, verbose: bool, remove_zip: bool) -> None:
    """Extracts ZIP archive to target directory and optionally removes the ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    if remove_zip:
        os.remove(zip_path)
        vprint("Zip removed successfully")

    vprint(f"File extracted to {extract_dir}")

def download_and_extract(url: str,
    zip_path: str,
    extract_dir: str,
    verbose: bool,
    remove_zip: bool
) -> bool:
    """Downloads and extracts dataset from Google Drive, returns True on success."""
    try:
        os.makedirs(extract_dir, exist_ok=True)
        
        download(url, zip_path, verbose)
        extract(zip_path, extract_dir, verbose, remove_zip)

        vprint(f"Dataset was downloaded and extracted to {extract_dir}.")
    except KeyboardInterrupt:
        print("Download Aborted.")
        return False
    return True
