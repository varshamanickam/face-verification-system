import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import json
import os


# Load the LFW dataset and print its info
ds, info = tfds.load(
    "lfw",
    split="train",      # LFW only has a single "train" split
    shuffle_files=False, # deterministic ingest
    with_info=True
)

print(info)

# Convert the dataset to a list of (image, label) tuples
#in order to record metadata
data = []
for example in tfds.as_numpy(ds):
    img = example["image"]
    label = example["label"]
    data.append((img, label))

# Extract labels and count occurrences of each label
labels = []
for example in tfds.as_numpy(ds):
    label = example["label"]
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    labels.append(label)

unique_labels, counts = np.unique(labels, return_counts=True)
label_counts = {str(k): int(v) for k, v in zip(unique_labels, counts)}

# Create a JSONmanifest with dataset metadata
manifest = {
    "dataset_name": "lfw",
    "num_images": len(labels),
    "num_identities": len(unique_labels),
    "label_counts": label_counts,
    "seed": 42,
    "split_policy": "none",  # we will split later
}

with open("manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Manifest written to manifest.json")