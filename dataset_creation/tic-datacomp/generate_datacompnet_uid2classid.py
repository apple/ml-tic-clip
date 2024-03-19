#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
import argparse
import pickle
import os
from subprocess import call
import json


tic_datacompnet_url = "https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacompnet_year2uids.pkl"
imagenet_class_info_url = "https://raw.githubusercontent.com/MadryLab/BREEDS-Benchmarks/master/imagenet_class_hierarchy/modified/dataset_class_info.json"
classnames_url = "https://huggingface.co/datasets/djghosh/wds_imagenet1k_test/raw/main/classnames.txt"


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


def main(args):
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

    # Write classnames.txt
    call(f"wget -nc {classnames_url} -P {output_path}", shell=True)
    # For verification only
    # See: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    with open(f"{output_path}/classnames.txt", "r") as f:
        for cls_id, cls_txt0 in enumerate(f):
            map_cls_txt = classid2name[cls_id].split(",")[0].lower()
            cls_txt = cls_txt0.rstrip().split(" ")[0].lower()
            # The names do not always match exactly, we test for partial match
            if cls_txt not in map_cls_txt:
                if False:
                    print(
                        f"{cls_id}/{synsets[cls_id]}: Class names do not match "
                        f" {cls_txt0.rstrip()} != {classid2name[cls_id]}."
                        " Using class name used in DataComp evals taken from CLIP."
                    )

    # Download tic-datacompnet year -> uid -> synset mappings
    call(f"wget -nc {tic_datacompnet_url} -P {output_path}", shell=True)
    with open(f"{output_path}/tic-datacompnet_year2uids.pkl", "rb") as f:
        year2uids = pickle.load(f)
    # Create uid -> classid mapping for given year
    os.makedirs(f"{output_path}/uid2classid/", exist_ok=True)
    for year, synset_uids in year2uids.items():
        uid2classid = {
            uid: synset2classid[synset]
            for synset, uids in year2uids[year].items()
            for uid in uids
        }
        with open(f"{output_path}/uid2classid/{year}.pkl", "wb") as f:
            pickle.dump(uid2classid, f)
        print(f"Created {output_path}/uid2classid/{year}.pkl: {len(uid2classid)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
