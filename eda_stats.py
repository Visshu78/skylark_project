import json
from collections import Counter

with open("train_dataset/gcp_marks.json", "r") as f:
    data = json.load(f)

print("="*50)
print("TOTAL LABELLED IMAGES")
print(len(data))

print("\n" + "="*50)
print("SHAPE DISTRIBUTION")

shape_counts = Counter(
    item.get("verified_shape", "MISSING")
    for item in data.values()
)

for shape, count in shape_counts.items():
    print(shape, count)

print("\n" + "="*50)
print("PROJECT COUNT")

projects = set()

for path in data.keys():
    project = path.split("/")[0]
    projects.add(project)

print(len(projects))

print("\nProjects:")
for p in sorted(projects):
    print("-", p)
