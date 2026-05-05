import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path("/home/beyza/image-forgery-detection")
RAW = BASE / "dataset" / "CASIA2"
OUT = BASE / "dataset" / "processed"

authentic_dir = RAW / "Au"
tampered_dir = RAW / "Tp"

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_images(folder):
    return [p for p in folder.iterdir() if p.suffix.lower() in valid_ext]

def split_and_copy(files, label):
    train, temp = train_test_split(files, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    groups = {
        "train": train,
        "val": val,
        "test": test
    }

    for split_name, split_files in groups.items():
        target = OUT / split_name / label
        target.mkdir(parents=True, exist_ok=True)

        for file in split_files:
            shutil.copy2(file, target / file.name)

    print(label)
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))

def main():
    authentic = get_images(authentic_dir)
    tampered = get_images(tampered_dir)

    print("Authentic total:", len(authentic))
    print("Tampered total:", len(tampered))

    split_and_copy(authentic, "authentic")
    split_and_copy(tampered, "tampered")

if __name__ == "__main__":
    main()
