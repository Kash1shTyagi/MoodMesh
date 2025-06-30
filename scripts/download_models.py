import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import hashlib
import zipfile
import sys

MODELS = {
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "files": [
            "det_10g.onnx",
            "w600k_r50.onnx",
            "genderage.onnx",
            "2d106det.onnx"
        ]
    },
    "affectnet_emotion": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "sha256": "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
    }
}

def download_file(url: str, dest: Path, chunk_size=1024):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            desc=f"Downloading {dest.name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def verify_checksum(file_path: Path, expected_sha: str) -> bool:
    """Verify file SHA256 checksum"""
    if not expected_sha:
        print(f"Skipping checksum for {file_path.name}")
        return True
        
    print(f"Verifying checksum for {file_path.name}...")
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    
    actual_sha = sha256.hexdigest()
    if actual_sha == expected_sha:
        print("Checksum verified")
        return True
        
    print(f"Checksum mismatch!\nExpected: {expected_sha}\nActual:   {actual_sha}")
    return False

def unzip_file(zip_path: Path, extract_to: Path):
    """Unzip a file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        zip_path.unlink()
        return True
    except Exception as e:
        print(f"Failed to unzip {zip_path.name}: {e}")
        return False

def main(models_dir: str, models: list):
    base_dir = Path(models_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
            
        model_info = MODELS[model_name]
        
        if model_name == "buffalo_l":
            zip_path = base_dir / "buffalo_l.zip"
            extracted_dir = base_dir / "buffalo_l"
            
            extracted_dir.mkdir(exist_ok=True)
            
            all_files_exist = all(
                (extracted_dir / file).exists() for file in model_info["files"]
            )
            
            if all_files_exist:
                print(f"InsightFace Buffalo_L model pack already exists")
                continue
                
            if not zip_path.exists():
                print(f"Downloading InsightFace Buffalo_L model pack...")
                if not download_file(model_info["url"], zip_path):
                    print("Failed to download Buffalo_L model pack")
                    continue
            
            print("Extracting InsightFace Buffalo_L model pack...")
            if unzip_file(zip_path, extracted_dir):
                print("Successfully extracted InsightFace Buffalo_L model pack")
            else:
                print("Failed to extract model pack")
                continue
                
            missing_files = [
                file for file in model_info["files"] 
                if not (extracted_dir / file).exists()
            ]
            
            if missing_files:
                print(f"Missing files after extraction: {', '.join(missing_files)}")
            else:
                print("All files extracted successfully")
            continue
        
        dest_path = base_dir / f"{model_name}.onnx"
        
        if dest_path.exists():
            if "sha256" in model_info:
                if verify_checksum(dest_path, model_info["sha256"]):
                    print(f"{model_name} already exists and valid")
                    continue
                else:
                    print(f"Invalid checksum for {model_name}, re-downloading")
                    dest_path.unlink()
            else:
                print(f"{model_name} already exists (no checksum verification)")
                continue
        
        print(f"Downloading {model_name}...")
        if not download_file(model_info["url"], dest_path):
            print(f"Failed to download {model_name}")
            continue
        
        if "sha256" in model_info:
            if not verify_checksum(dest_path, model_info["sha256"]):
                print(f"Checksum verification failed for {model_name}")
                if input("Delete invalid file? (y/n): ").lower() == 'y':
                    dest_path.unlink()
        else:
            print(f"No checksum provided for {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download emotion AI models")
    parser.add_argument("--models-dir", default="data/models", help="Models directory")
    parser.add_argument("--models", nargs="+", 
                        default=["buffalo_l", "affectnet_emotion"],
                        help="Models to download")
    args = parser.parse_args()
    
    main(args.models_dir, args.models)