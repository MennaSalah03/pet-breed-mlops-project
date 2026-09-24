import json
import os
import re

import config
from PIL import Image

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CORRUPT_DIR = os.path.join(DATA_DIR, "corrupted_images")
OUT = os.path.join(DATA_DIR, "manifest.json")


def breed_of(image_id):
    return re.sub(r"_\d+$", "", image_id).replace("_", " ")


def img_size(path):
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except FileNotFoundError:
        return None, None


def parse_list(path, split, class_lookup):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id, species_id, _breed_id = line.split()
            class_lookup[image_id] = int(class_id) - 1
            p = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
            w, h = img_size(p)
            entries.append({
                "image_id": image_id,
                "path": p,
                "breed": breed_of(image_id),
                "species": config.SPECIES[species_id],
                "class_index": class_lookup[image_id],
                "split": split,
                "corruption": None,
                "severity": 0,
                "width": w,
                "height": h,
            })
    return entries


def parse_corrupted(class_lookup):
    entries = []
    if not os.path.isdir(CORRUPT_DIR):
        return entries
    for fname in os.listdir(CORRUPT_DIR):
        m = config.CORRUPT_RE.match(fname)
        if not m:
            continue
        image_id, corr, sev = m.group("id"), m.group("corr"), int(m.group("sev"))
        p = os.path.join(CORRUPT_DIR, fname)
        w, h = img_size(p)
        entries.append({
            "image_id": image_id,
            "path": p,
            "breed": breed_of(image_id),
            "species": "cat" if image_id[0].isupper() else "dog",
            "class_index": class_lookup.get(image_id),
            "split": "test",
            "corruption": corr,
            "severity": sev,
            "width": w,
            "height": h,
        })
    return entries


def main():
    class_lookup = {}
    manifest = parse_list(config.TRAIN_LIST, "train", class_lookup)
    manifest += parse_list(config.TEST_LIST, "test", class_lookup)
    manifest += parse_corrupted(class_lookup)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} entries to {OUT}")


if __name__ == "__main__":
    main()