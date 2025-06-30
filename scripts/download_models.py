import os
import sys
import requests
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm

# === UPDATED: point at the validated/ folder, not raw/main ===
MODELS = {
    "retinaface": {
        "url": "https://raw.githubusercontent.com/discipleofhamilton/RetinaFace/master/FaceDetector.onnx",
        "sha256": None
    },
    "emotion_ferplus": {
        "url": "https://huggingface.co/webai-community/models/resolve/main/emotion-ferplus-8.onnx",
        "sha256": "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
    }
}


def download_file(url: str, dest: Path, expected_sha: str = None) -> bool:
    """Download a file with progress bar and optional SHA256 verification"""
    print(f"→ Downloading {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        chunk = 1 << 20  # 1 MB

        with open(dest, "wb") as fp, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for buf in response.iter_content(chunk_size=chunk):
                if buf:
                    fp.write(buf)
                    bar.update(len(buf))

        if expected_sha:
            actual = sha256_of_file(dest)
            if actual.lower() != expected_sha.lower():
                print(f"!! checksum mismatch: {actual} != {expected_sha}")
                dest.unlink()
                return False

        return True

    except Exception as e:
        print(f"!! download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def main(models_dir: str, models: list):
    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in models:
        cfg = MODELS.get(name)
        if not cfg:
            print(f"Unknown model '{name}'  → skipping")
            continue

        dest = out / f"{name}.onnx"
        
        if dest.exists() and cfg.get("sha256"):
            if sha256_of_file(dest).lower() == cfg["sha256"].lower():
                print(f"{name}: already present & valid")
                continue
            print(f"{name}: bad checksum → re-downloading")

        if download_file(cfg["url"], dest, cfg.get("sha256")):
            print(f"{name}: ✓ saved to {dest}")
        else:
            print(f"{name}: ✗ failed")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="data/models")
    p.add_argument(
        "--models",
        nargs="+",
        default=["retinaface", "emotion_ferplus"]
    )
    args = p.parse_args()
    main(args.models_dir, args.models)
