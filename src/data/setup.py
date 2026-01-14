import gdown
import zipfile
import os

VERBOSE = True


def vprint(verbose: bool, *args, **kwargs) -> None:
    """Prints only if verbose is True."""
    if verbose:
        print(*args, **kwargs)


def download(url: str, zip_path: str, verbose: bool) -> None:
    """Downloads file from Google Drive URL to specified path."""
    gdown.download(url, zip_path, quiet=False)

    if not os.path.exists(zip_path):
        vprint(verbose, "The dataset could not be downloaded sucessfully.")
        vprint(verbose, "Please download it manully using the url below and place it in datasets/anti-uav300-raw.\n")

        vprint(verbose, "-" * 50)
        vprint(verbose, url)
        vprint(verbose, "-" * 50)

        input("\nPress Enter to continue.")
    else:
        vprint(verbose, f"Dataset download completed.")

def extract(zip_path: str, extract_dir: str, verbose: bool, remove_zip: bool) -> None:
    """Extracts ZIP archive to target directory and optionally removes the ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    if remove_zip:
        os.remove(zip_path)
        vprint(verbose, "Zip removed successfully")

    vprint(verbose, f"File extracted to {extract_dir}")

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

        vprint(verbose, f"Dataset was downloaded and extracted to {extract_dir}.")
    except KeyboardInterrupt:
        print("Download Aborted.")
        return False
    return True
