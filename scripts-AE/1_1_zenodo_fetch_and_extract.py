#!/usr/bin/env python3
"""
Download all files from a Zenodo record and extract .tar.gz archives,
stripping the absolute paths stored inside them.

Usage:
    python zenodo_fetch_and_extract.py --record_id RECORD_ID --target_dir TARGET_DIR [--token TOKEN]

Example:
    python zenodo_fetch_and_extract.py --record_id 15300098 --target_dir ./data
"""

import argparse
import hashlib
import os
import sys
import time
import tarfile
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import requests
import subprocess

def get_record_json(record_id: str, token: str | None = None) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_files(record_json: dict, token: str | None = None):
    files = record_json.get("files", [])
    if files:
        return files

    # try to fetch files from the "links" section
    files_url = record_json.get("links", {}).get("files")
    print(f"Fetching files from {files_url} …")
    if files_url:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(files_url, headers=headers, timeout=30)
        if r.status_code == 403:
            raise RuntimeError(
                "Access denied: your account or secret link "
                "is not authorized to view these restricted files."
            )
        r.raise_for_status()
        files = r.json()
        if files:
            return files

    if record_json.get("metadata", {}).get("access_right") == "restricted":
        raise RuntimeError(
            "This record's files are restricted. Ask the owner "
            "to share the record with your user or a secret link, "
            "then retry with the same token."
        )
    raise RuntimeError("No files found in record metadata!")


def md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def download(url: str, dest: Path, token: str | None = None,
             retries: int = 9999, delay: int = 10) -> None:
    """
    Download a file with wget, auto-resuming and infinite retry.
    - retries: maximum outer-loop retries (keep 9 999 to match old code).
    - delay  : seconds to sleep after a retryable failure.
    """
    if not shutil.which("wget"):
        raise RuntimeError("wget not found, please install it")

    # Build the static part of the command only once
    cmd = [
        "wget",
        "--continue",                 # resume partial file (-c)
        "--tries=0",                  # infinite internal retries
        "--waitretry=10",             # linear back-off up to 10 s
        "--retry-connrefused",        # treat ECONNREFUSED as retryable
        "--read-timeout=30",          # abort hung connections faster
        "-O", str(dest),              # output path
    ]
    if token:
        cmd += [f'--header=Authorization: Bearer {token}']
    cmd.append(url)

    attempt = 0
    while attempt < retries:
        attempt += 1
        print(f"  • wget attempt {attempt}: {' '.join(cmd)}")
        ec = subprocess.call(cmd)
        if ec == 0:                 # success
            break
        # Retryable wget exit codes: 4 = network failure, 8 = server error
        if ec not in (4, 8):
            raise RuntimeError(f"wget failed with exit code {ec}")
        print(f"    partial file, retrying in {delay} s … (wget exit {ec})")
        time.sleep(delay)
    else:
        raise RuntimeError(f"wget gave up after {retries} retries")

def safe_extract_tar(tar_path: Path, target_dir: Path) -> None:
    """Extract tar.gz, removing leading '/' and forbidding path traversal."""
    def sanitize(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        # Drop leading '/' to remove absolute paths
        tarinfo.name = tarinfo.name.lstrip("/")
        # Resolve final path and check it stays inside target_dir
        final_path = target_dir / tarinfo.name
        if not final_path.resolve().is_relative_to(target_dir.resolve()):
            raise RuntimeError(f"Unsafe path detected in archive: {tarinfo.name}")
        return tarinfo

    before = {p.name for p in target_dir.iterdir()}

    with tarfile.open(tar_path, "r:gz") as tar:
        kw = {"filter": "fully_trusted"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(path=target_dir, members=(sanitize(m) for m in tar.getmembers()), **kw)

    # Fix the issue of uploading with paths starting from the root directory.
    # If the tar contains a single directory, move its contents up one level
    # to avoid nested directories.
    def _recursive_flatten(wrapper: Path, root: Path) -> None:
        """Flatten a directory structure by moving the leaf directory up one level."""
        leaf = wrapper
        while True:
            entries = list(leaf.iterdir())
            dirs   = [d for d in entries if d.is_dir()]
            files  = [f for f in entries if not f.is_dir()]
            if files or len(dirs) != 1:
                break
            leaf = dirs[0]
        if leaf == wrapper:
            return
        dest = root / leaf.name
        if dest.exists():
            raise RuntimeError(f"Cannot flatten: target {dest} already exists")
        shutil.move(str(leaf), dest)
        shutil.rmtree(wrapper)
        print(f"Flattened {wrapper} → {dest}")

    added = [p for p in target_dir.iterdir() if p.name not in before]
    print(f"Added {len(added)} new items to {target_dir}: {[p.name for p in added]}")

    tar_path_stem = tar_path
    while tar_path_stem.suffix:
        tar_path_stem = tar_path_stem.with_suffix("")

    print(f"Checking if {tar_path_stem} is a single directory …")
    if len(added) == 1 and added[0].is_dir() and added[0].name != tar_path_stem.name:
        _recursive_flatten(added[0], target_dir)

def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract .zip safely, strip absolute paths and path traversal attempts."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            name = member.lstrip("/\\")
            dest = target_dir / name
            dest_path = dest.resolve()
            if not str(dest_path).startswith(str(target_dir.resolve()) + os.sep):
                raise RuntimeError(f"Unsafe path in zip: {member}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if member.endswith("/"):
                dest.mkdir(exist_ok=True)
            else:
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

def main() -> None:
    parser = argparse.ArgumentParser(description="Download & extract Zenodo files.")
    parser.add_argument("--record_id", help="Zenodo record ID (e.g. 15300098)")
    parser.add_argument("--target_dir", type=Path, help="Directory to put the data in")
    parser.add_argument(
        "--token",
        default=os.getenv("ZENODO_TOKEN"),
        help="Zenodo personal access token (or set ZENODO_TOKEN env var)",
    )
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching record {args.record_id} metadata …")
    record = get_record_json(args.record_id, args.token)

    for fmeta in iter_files(record, args.token):
        print(f"Processing file: {fmeta['key']}")

        fname = fmeta["key"]
        links = fmeta.get("links", {})
        url = links.get("download") or links.get("content") or links.get("self")
        print(f"  • Download URL: {url}")
        if not url:
            raise RuntimeError(f"No usable download URL found for {fname}")
        checksum = fmeta.get("checksum", "")
        md5_expected = checksum.split(":")[1] if checksum.startswith("md5:") else None
        outfile = args.target_dir / fname
        if outfile.exists():
            print(f"✔ {fname} already exists, skipping download. Please remove it if you want to re-download.")
        else:
            print(f"↓ Downloading {fname} → {outfile}")
            download(url, outfile, args.token)

        if md5_expected:
            print(f"  • Verifying MD5 …", end="", flush=True)
            if md5(outfile) != md5_expected:
                raise RuntimeError(f"\nMD5 mismatch for {fname}")
            print(" OK")

        if fname.endswith(".tar.gz"):
            print(f"  • Extracting {fname} …")
            safe_extract_tar(outfile, args.target_dir)
        elif fname.endswith(".zip"):
            print(f"  • Extracting {fname} …")
            safe_extract_zip(outfile, args.target_dir)
        else:
            print(f"  • Skipping extraction for {fname} (not a .tar.gz or .zip file)")
        
        print(f"✔ Processed {fname}")
    print("\nAll done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
