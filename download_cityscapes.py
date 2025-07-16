import zipfile
import os
import argparse
import requests
from tqdm import tqdm


FILE_URLS = [
    "https://www.cityscapes-dataset.com/file-handling/?packageID=1",  ## gtFine_trainvaltest.zip (241MB)
    "https://www.cityscapes-dataset.com/file-handling/?packageID=3",  ## leftImg8bit_trainvaltest.zip (11GB)
]


def download_file(url: str, out_dir: str, cookie: str):

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (YOUR-USER-AGENT-HERE)",
        "Cookie": cookie  ## "PHPSESSID=vi6m4070fpbe6sfjb5i1sm1v6e"
    }

    local_filename = os.path.join(out_dir, url.split("packageID=")[-1] + ".zip")
    with requests.get(url, headers=HEADERS, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded: {local_filename}")


def unzip_files(out_dir: str):
    """Unzip all .zip files in the given directory."""
    for filename in os.listdir(out_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(out_dir, filename)
            extract_path = os.path.join(out_dir, filename[:-4])
            print(f"Unzipping {zip_path} to {extract_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Unzipped: {zip_path}")
            try:
                os.remove(zip_path)
                print(f"Deleted: {zip_path}")
            except Exception as e:
                print(f"Failed to delete {zip_path}: {e}")
    print("Unzipped all files.")


if __name__ == "__main__":

    help_cookie = """
    Use Chrome as browser:
    1.	Log in to https://www.cityscapes-dataset.com/downloads/.
	2.	Right-click anywhere on the page → Click Inspect (or press Cmd + Option + I).
	3.	Go to the Network tab.
	4.	Click to download one of the dataset files (just start it; cancel if needed).
	5.	Find the request in the Network tab (it will be named like file-handling?packageID=...).
	6.	Click it and go to the Headers tab.
	7.	Scroll down to Request Headers.
	8.	Look for the line that starts with 'Cookie'
    """
    help_package = """
    Position of zip file on page: https://www.cityscapes-dataset.com/downloads/ 
    (Copy the the link of the file to see)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cookie",
        type=str,
        required=True,
        help=help_cookie
    )
    parser.add_argument(
        "-out", "--out_dir", 
        type=str, 
        default="cityscapes",
        help="Output directory to save data."
    )
    parser.add_argument(
        "-ids", "--package_ids", 
        type=int, 
        nargs="+",
        choices=[i for i in range(1, 34)],
        default=[1, 3],
        help=help_package
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    file_urls = [f"https://www.cityscapes-dataset.com/file-handling/?packageID={i}" for i in list(args.package_ids)]

    """
    for url in file_urls:
        if url not in os.listdir(out_dir):
            try:
                download_file(url, out_dir, args.cookie)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
        else:
            print(f"file: {url} already exists in {out_dir}")

    print("Download completed!\nExtracting...\n")

    """
    unzip_files(out_dir)

    print("Done!")
