#!/usr/bin/env python3
"""Download and prepare benchmark datasets for SupeRANSAC evaluation.

Datasets supported:
1. magsac_data  : kusvod2, AdelaideRMF, Multi-H, homogr, EVD (Extreme View Dataset), Strecha
2. phototourism : CVPR 2020 RANSAC Tutorial (Palace of Westminster / Sacre Coeur)
3. eth3d        : ETH3D multi-view high-res undistorted DSLR scenes
4. 7scenes      : Microsoft 7Scenes RGB-D indoor localization/pose benchmark

Usage:
    python tests/download_datasets.py --data_dir ./data --dataset all
"""

import argparse
import glob
import os
import shutil
import subprocess
import urllib.request
import zipfile


ETH3D_SCENES = [
    "courtyard",
    "electro",
    "facade",
    "kicker",
    "office",
    "playground",
    "relief",
]

SEVEN_SCENES_LIST = [
    "heads",
    "fire",
    "chess",
]


def download_url(url, dst_path):
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
        print(f"  [cached] {dst_path}")
        return True
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    print(f"  Downloading {url} -> {dst_path} ...")
    cmd = ["curl", "-L", "--fail", "--retry", "3", "-o", dst_path, url]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False
    return True


def setup_magsac_data(data_dir):
    target_dir = os.path.join(data_dir, "magsac_data")
    if os.path.exists(os.path.join(target_dir, "fundamental_matrix")) and os.path.exists(
        os.path.join(target_dir, "homography")
    ):
        print("[magsac_data] Already present at", target_dir)
        return
    local_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "magsac_data", "data"))
    if os.path.exists(local_candidate):
        if os.path.lexists(target_dir):
            os.remove(target_dir)
        os.symlink(local_candidate, target_dir)
        print("[magsac_data] Linked local magsac_data ->", target_dir)
        return
    tmp_repo = os.path.join(data_dir, "_tmp_magsac_repo")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/danini/magsac.git", tmp_repo], check=True)
    shutil.move(os.path.join(tmp_repo, "data"), target_dir)
    shutil.rmtree(tmp_repo, ignore_errors=True)
    print("[magsac_data] Ready at", target_dir)


def setup_phototourism(data_dir):
    pt_dir = os.path.join(data_dir, "phototourism")
    scene_dir = os.path.join(pt_dir, "val", "palace_of_westminster")
    if os.path.exists(os.path.join(scene_dir, "Fgt.h5")) and os.path.isdir(os.path.join(scene_dir, "images")):
        print("[phototourism] Already present at", scene_dir)
        return
    os.makedirs(scene_dir, exist_ok=True)
    url = "http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar"
    print("[phototourism] Fetching PhotoTourism scene via HTTP Range:", url)

    req_head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req_head, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    def fetch_range(start, length):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    offset = 0
    while offset + 512 <= total_size:
        hdr = fetch_range(offset, 512)
        if len(hdr) < 512 or hdr == b"\x00" * 512:
            break
        raw_name = hdr[0:100].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        raw_prefix = hdr[345:500].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        name = f"{raw_prefix}/{raw_name}" if raw_prefix else raw_name
        size_str = hdr[124:136].split(b"\x00", 1)[0].strip()
        size = int(size_str, 8) if size_str else 0
        typeflag = hdr[156:157]
        data_offset = offset + 512

        if name.startswith("RANSAC-Tutorial-Data/train/palace_of_westminster/"):
            sub = name[len("RANSAC-Tutorial-Data/train/palace_of_westminster/") :]
            if sub in ("Fgt.h5", "Egt.h5", "T.h5", "K1_K2.h5", "R.h5"):
                out_file = os.path.join(scene_dir, sub)
                if not os.path.exists(out_file) or os.path.getsize(out_file) != size:
                    print(f"  [phototourism] Downloading {sub} ({size / 1e6:.2f} MB) ...")
                    subprocess.run(
                        ["curl", "-s", "-L", "--fail", "--retry", "3", "-r", f"{data_offset}-{data_offset + size - 1}", "-o", out_file, url],
                        check=True,
                    )
            elif sub == "images/":
                print("  [phototourism] Streaming images/ archive slice (~350 MB) ...")
                slice_len = 360 * 1024 * 1024
                cmd = (
                    f"curl -s -L -r {offset}-{offset + slice_len - 1} {url} "
                    f"| tar -xf - -C {scene_dir} --strip-components=3 RANSAC-Tutorial-Data/train/palace_of_westminster/images || true"
                )
                subprocess.run(["bash", "-c", cmd], check=False)
                break

        offset = data_offset + ((size + 511) // 512) * 512
    print("[phototourism] Ready at", scene_dir)


def setup_eth3d(data_dir, scenes=None):
    eth_dir = os.path.join(data_dir, "eth3d")
    os.makedirs(eth_dir, exist_ok=True)
    scenes = scenes or ETH3D_SCENES
    for sc in scenes:
        sc_dir = os.path.join(eth_dir, sc)
        if os.path.exists(os.path.join(sc_dir, "dslr_calibration_undistorted", "cameras.txt")):
            print(f"[eth3d:{sc}] Already present")
            continue
        archive_path = os.path.join(eth_dir, f"{sc}_dslr_undistorted.7z")
        url = f"https://www.eth3d.net/data/{sc}_dslr_undistorted.7z"
        if not download_url(url, archive_path):
            print(f"[eth3d:{sc}] Warning: Failed to download {url}")
            continue
        print(f"[eth3d:{sc}] Extracting {archive_path} ...")
        subprocess.run(["7z", "x", "-y", f"-o{eth_dir}", archive_path], check=True)
        os.remove(archive_path)
    print("[eth3d] Ready at", eth_dir)


def setup_7scenes(data_dir, scenes=None):
    s7_dir = os.path.join(data_dir, "7scenes")
    os.makedirs(s7_dir, exist_ok=True)
    scenes = scenes or ["heads", "fire", "chess"]
    for sc in scenes:
        sc_dir = os.path.join(s7_dir, sc)
        if os.path.isdir(sc_dir):
            for seq_zip in os.listdir(sc_dir):
                if seq_zip.endswith(".zip"):
                    seq_zip_path = os.path.join(sc_dir, seq_zip)
                    subprocess.run(["7z", "x", "-y", f"-o{sc_dir}", seq_zip_path], check=False)
                    os.remove(seq_zip_path)
            if any(k.startswith("seq-") and os.path.isdir(os.path.join(sc_dir, k)) for k in os.listdir(sc_dir)):
                print(f"[7scenes:{sc}] Already present")
                continue
        zip_path = os.path.join(s7_dir, f"{sc}.zip")
        url = f"http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/{sc}.zip"
        if not download_url(url, zip_path):
            print(f"[7scenes:{sc}] Warning: Failed to download {url}")
            continue
        print(f"[7scenes:{sc}] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(s7_dir)
        os.remove(zip_path)
        for seq_zip in os.listdir(sc_dir):
            if seq_zip.endswith(".zip"):
                seq_zip_path = os.path.join(sc_dir, seq_zip)
                subprocess.run(["7z", "x", "-y", f"-o{sc_dir}", seq_zip_path], check=False)
                os.remove(seq_zip_path)
    print("[7scenes] Ready at", s7_dir)


def setup_hpatches(data_dir):
    """Optional: HPatches sequences (580 viewpoint/illumination pairs with GT homographies)."""
    import tarfile

    hp_dir = os.path.join(data_dir, "hpatches")
    if os.path.isdir(hp_dir) and len(glob.glob(os.path.join(hp_dir, "v_*"))) >= 50:
        print("[hpatches] Already present at", hp_dir)
        return
    tar_path = os.path.join(data_dir, "hpatches-sequences-release.tar.gz")
    urls = [
        "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz",
        "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.tar.gz",
    ]
    if not any(download_url(u, tar_path) for u in urls):
        print("[hpatches] Optional dataset not reachable from this machine -- skipping (homography track uses Homogr + EVD).")
        return
    os.makedirs(hp_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(hp_dir)
    nested = os.path.join(hp_dir, "hpatches-sequences-release")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), os.path.join(hp_dir, item))
        os.rmdir(nested)
    os.remove(tar_path)
    print("[hpatches] Ready at", hp_dir)


def verify_6d_examples():
    ex_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "data")
    required = [
        "pose6dscene_points.txt",
        "pose6dscene_gt.txt",
        "pose6dscene.K",
        "rigid_pose_example_points.txt",
        "rigid_pose_example_gt.txt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ex_dir, f))]
    if not missing:
        print("[6d_examples] Ready at", ex_dir)
    else:
        print(f"[6d_examples] Warning: Missing files in {ex_dir}: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets for SupeRANSAC")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Target directory where benchmark datasets will be stored.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        default=["all"],
        choices=["all", "magsac_data", "phototourism", "eth3d", "7scenes", "hpatches"],
        help="Which datasets to download.",
    )
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    selected = set(args.datasets)
    if "all" in selected:
        selected = {"magsac_data", "phototourism", "eth3d", "7scenes", "hpatches"}

    if "magsac_data" in selected:
        setup_magsac_data(args.data_dir)
    if "phototourism" in selected:
        setup_phototourism(args.data_dir)
    if "eth3d" in selected:
        setup_eth3d(args.data_dir)
    if "7scenes" in selected:
        setup_7scenes(args.data_dir)
    if "hpatches" in selected:
        setup_hpatches(args.data_dir)
    verify_6d_examples()


if __name__ == "__main__":
    main()
