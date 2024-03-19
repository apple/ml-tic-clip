#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
import argparse
import numpy as np
import pickle
import os
from subprocess import call


def get_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the local .npy files to be saved.",
    )
    args = parser.parse_args()
    return args


def uids_str_to_np(uids):
    processed_uids = np.array(
        [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], np.dtype("u8,u8")
    )
    processed_uids.sort()
    processed_uids = np.unique(processed_uids)
    return processed_uids


url_basepath = "https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/"
uids_fname = [
    "tic-datacomp_retrieval_evals_year2uids.pkl",
    "tic-datacompnet_year2uids.pkl",
]


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(f"{args.output_path}/eval/retrieval/", exist_ok=True)
    os.makedirs(f"{args.output_path}/eval/datacompnet/", exist_ok=True)

    for fname in uids_fname:
        call(f"wget -nc {url_basepath}{fname} -P {args.output_path}", shell=True)

    # Process tic-datacomp-retrieval
    all_uids = []
    with open(f"{args.output_path}/{uids_fname[0]}", "rb") as f:
        year2uids = pickle.load(f)
    count = 0
    for year, uids in year2uids.items():
        uids_np = uids_str_to_np(uids)
        count += len(uids_np)
        np.save(f"{args.output_path}/eval/retrieval/{year}.npy", uids_np)
    print(f"Created {args.output_path}/eval/retrieval/<year>.npy: {count}")
    all_uids += [uid for uids in year2uids.values() for uid in uids]

    # Process tic-datacompnet
    with open(f"{args.output_path}/{uids_fname[1]}", "rb") as f:
        year2uids = pickle.load(f)
    count = 0
    for year, synset_uids in year2uids.items():
        uids = [uid for synset, uids in synset_uids.items() for uid in uids]
        uids_np = uids_str_to_np(uids)
        count += len(uids_np)
        np.save(f"{args.output_path}/eval/datacompnet/{year}.npy", uids_np)
    print(f"Created {args.output_path}/eval/datacompnet/<year>.npy: {count}")
    all_uids += [
        uid
        for synset_uids in year2uids.values()
        for uids in synset_uids.values()
        for uid in uids
    ]
    all_uids_np = uids_str_to_np(all_uids)
    np.save(f"{args.output_path}/eval/alluids.npy", all_uids_np)
    print(f"Created {args.output_path}/alluids.npy: {len(all_uids_np)}")
