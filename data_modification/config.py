import os
import re

DATA_DIR = "data/oxford-iiit-pet"
TEST_LIST = os.path.join(DATA_DIR, "annotations/test.txt")
TRAIN_LIST = os.path.join(DATA_DIR, "annotations/trainval.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUT = os.path.join(DATA_DIR, "manifest.json")
CORRUPT_DIR=os.path.join(DATA_DIR, "corrupted_images")
SEVERITIES = [1, 2, 3, 4, 5]

SPECIES = {"1": "cat", "2": "dog"}
CORRUPT_RE = re.compile(r"^(?P<id>.+)_(?P<corr>[a-z_]+)_(?P<sev>\d)\.jpg$")
