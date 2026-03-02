import json
import os

import numpy as np

lfw_data_dir = "data/lfw"

# if the dataset is not already downloaded, it will be downloaded to data_cache/downloads 
# and extracted to data_cache/downloads/extracted then copied to data/lfw. 
if not os.path.exists(lfw_data_dir):
    import tensorflow_datasets as tfds

    # Load the LFW dataset and print its info
    ds, info = tfds.load(
        "lfw",
        split="train",      # LFW only has a single "train" split
        shuffle_files=False, # deterministic ingest
        with_info=True,
        data_dir="data_cache"
    )

    # find the extracted data directory
    # it should look something like
    # data_cache/downloads/extracted/TAR_GZ.ndownloader.figshare.com_files_5976018BV99nGMtc3Dm-0r8dGjUD5cMNKgNTG9Q_-xj9ajVNsA/lfw
    extracted_data_dir = None
    for root, dirs, files in sorted(os.walk("data_cache/downloads/extracted")):
        for dir in dirs:
            if "TAR_GZ" in dir:
                # try finding the lfw directory inside this dir
                potential_lfw_dir = os.path.join(root, dir, "lfw")
                if os.path.isdir(potential_lfw_dir):
                    extracted_data_dir = potential_lfw_dir
                    break
        if extracted_data_dir:
            break
    print(f"Extracted data directory: {extracted_data_dir}")

    # copy entire directory to data/lfw
    import shutil
    if os.path.exists(lfw_data_dir):
        shutil.rmtree(lfw_data_dir)
    shutil.copytree(extracted_data_dir, lfw_data_dir)
    print(f"Copied data to {lfw_data_dir}")
    # remove the original extracted directory to save space
    shutil.rmtree(os.path.dirname(extracted_data_dir))
    print(f"Removed original extracted directory: {os.path.dirname(extracted_data_dir)}")


# create a list of (image_path, label) tuples
data = []
for label in sorted(os.listdir(lfw_data_dir)):
    label_dir = os.path.join(lfw_data_dir, label)
    if os.path.isdir(label_dir):
        for img_file in sorted(os.listdir(label_dir)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(label_dir, img_file)
                data.append((img_path, label))
                
print(f"Total samples found: {len(data)}")
print(f"Unique labels found: {len(set([label for _, label in data]))}")

# Split the dataset into training and validation sets and test sets, by identity, to avoid data leakage. 
# We will use 80/10/10 split for train/val/test.
# Use a fixed random seed for reproducibility
unique_labels = list(set([label for _, label in data]))
#sort by label name to ensure deterministic split
unique_labels.sort()
num_labels = len(unique_labels)
num_train = int(0.8 * num_labels)
num_val = int(0.1 * num_labels)
num_test = int(0.1 * num_labels)
train_labels = unique_labels[:num_train]
val_labels = unique_labels[num_train:num_train + num_val]
test_labels = unique_labels[num_train + num_val:num_train + num_val + num_test]

num_train_images = sum([1 for _, label in data if label in train_labels])
num_val_images = sum([1 for _, label in data if label in val_labels])
num_test_images = sum([1 for _, label in data if label in test_labels])

# set random seed for reproducibility
random_seed = 42

# Create a manifest json
manifest = {
    "seed": random_seed,
    "split_policy": {
        "split_basis": "identity level",
        "split_ratios":{"train": 0.8, "val": 0.1, "test": 0.1 },
        "method": "alphabetical sort and then slice",
        "description": "Dataset is split by identity (person) and not by image. This ensures that no identity appears in more than one split."

    },
    "counts": {
        "train_identities": len(train_labels),
        "val_identities": len(val_labels),
        "test_identities": len(test_labels),
        "train_images": num_train_images,
        "val_images": num_val_images,
        "test_images": num_test_images,

    },
    "data_source": "LFW dataset from TensorFlow Datasets at https://www.tensorflow.org/datasets/catalog/lfw, downloaded and extracted to data/lfw",
}

output_path = os.path.join("outputs", "manifest.json") 
os.makedirs(os.path.dirname(output_path), exist_ok=True) 
with open(output_path, "w") as f: json.dump(manifest, f, indent=2) 
print("Manifest written to outputs/manifest.json")


def generate_pairs(id_map: dict, num_pairs: int, seed: int):
    """
    Returns list of [i, j, y] pairs, y in {0,1}, roughly 50/50 pos/neg.
    Deterministic given seed.
    """
    rng = np.random.default_rng(seed)

    # We want roughly equal numbers of same and different pairs
    same_count = num_pairs // 2
    diff_count = num_pairs - same_count

    pairs = []
    # 1 for same identity
    i = 0
    while i < same_count:
        id = rng.choice(list(id_map.keys()))
        img1, img2 = rng.choice(id_map[id], size=2, replace=True) # allow identical pairs (same image) as well as different pairs of the same identity
        pairs.append([img1, img2, 1])
        i += 1

    # 0 for different identities
    i = 0
    while i < diff_count:
        id1, id2 = rng.choice(list(id_map.keys()), size=2, replace=False)
        img1 = rng.choice(id_map[id1])
        img2 = rng.choice(id_map[id2])
        pairs.append([img1, img2, 0])
        i += 1

    # Deterministic shuffle
    rng.shuffle(pairs)
    return pairs


# create train, val, test id_maps
train_id_map = {id: [img for img, label in data if label == id and label in train_labels] for id in train_labels}
val_id_map = {id: [img for img, label in data if label == id and label in val_labels] for id in val_labels}
test_id_map = {id: [img for img, label in data if label == id and label in test_labels] for id in test_labels}

# generate pairs for each split
train_pairs = generate_pairs(train_id_map, num_pairs=10000, seed=random_seed)
val_pairs = generate_pairs(val_id_map, num_pairs=2000, seed=random_seed)
test_pairs = generate_pairs(test_id_map, num_pairs=2000, seed=random_seed)

# save each split to a jsonl file with fields "left_path", "right_path", "label", "split"

def save_pairs(pairs, split):
    pairs_dir = os.path.join("outputs", "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    output_path = os.path.join(pairs_dir, f"{split}.jsonl")

    with open(output_path, "w") as f:
        for img1, img2, label in pairs:
            json_line = json.dumps({
                "left_path": img1,
                "right_path": img2,
                "label": label,
                "split": split
            })
            f.write(json_line + "\n")

    print(f"Saved {len(pairs)} pairs to {output_path}")

save_pairs(train_pairs, "train")
save_pairs(val_pairs, "val")
save_pairs(test_pairs, "test")