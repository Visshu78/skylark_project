from PIL import Image
import json
import os
from collections import Counter


with open("train_dataset/gcp_marks.json") as f:
    data = json.load(f)

first_path = next(iter(data.keys()))

img = Image.open(
    os.path.join("train_dataset", first_path)
)

print("Image Size:", img.size)

with open("train_dataset/gcp_marks.json") as f:
    data = json.load(f)

sizes = Counter()

for path in list(data.keys())[:100]:  # first 100 images
    img = Image.open(
        os.path.join("train_dataset", path)
    )
    sizes[img.size] += 1

print(sizes)