#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""This script groups imagenet synsets to calculate restricted ImageNet accuracy."""
import argparse
import os
from subprocess import call
import json
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")


imagenet_class_info_url = "https://raw.githubusercontent.com/MadryLab/BREEDS-Benchmarks/master/imagenet_class_hierarchy/modified/dataset_class_info.json"
nodes_restricted = [
    "device",
    "motor_vehicle",
    "machine",
    "phone",
    "computer",
    "garment",
    "clothing",
    "dog",
    "bird",
    "cat",
    "animal",
]


def get_all_child(root_synset):
    from collections import deque

    visited = set()

    queue = deque([root_synset])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            children = node.hyponyms()
            for child in children:
                if child not in visited:
                    queue.append(child)
    return visited


def get_intersection_imagenet(nodes, imagenet_map):
    class_idx = []
    for node in nodes:
        wordnet_id = "n{:08d}".format(node.offset())
        if wordnet_id in imagenet_map:
            class_idx.append(imagenet_map[wordnet_id])
    return class_idx


def get_imagenet_classes(root_name, synset2classid):
    root_node = wn.synsets(f"{root_name}")
    all_children = set()
    for node in root_node:
        all_children = all_children.union(get_all_child(node))
    ids = get_intersection_imagenet(all_children, synset2classid)
    return ids


def get_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local path to save <year>/uid2classid.pkl files.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Download the file containing list[[imagenet_class_id, wordnet_synset, class_name]]
    call(f"wget -nc {imagenet_class_info_url} -P {output_path}", shell=True)
    with open(f"{output_path}/dataset_class_info.json", "rb") as f:
        arr_list = json.loads(f.read().decode("utf-8"))
    synsets = []
    synset2classid = {}
    classid2name = {}
    for val in arr_list:
        synsets.append(val[1])  # wordnet_synset: "nxxxxxxxx"
        synset2classid[val[1]] = val[0]  # wordnet_synset -> imagenet_class_id (0-999)
        classid2name[val[0]] = val[2]  # imagenet_class_id -> class_name

    for node in nodes_restricted:
        ids = get_imagenet_classes(node, synset2classid)
        print(ids)
