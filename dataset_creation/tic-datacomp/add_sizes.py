#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""Generate {YEAR/MONTH}/sizes.json with the number of samples in each tar shard."""
import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import simdjson
import time
from dataset_creation.utils import fsspec_url_to_fs, exponential_backoff, list_timesplit_paths


def generate_json(url):
    """Generate one sizes.json for a single split."""
    fs, path = fsspec_url_to_fs(url)

    all_files = fs.ls(path)
    all_files = [f for f in all_files if "_stats.json" in f]

    curr_sizes = {}
    for file in all_files:
        tar_name = f"{os.path.basename(file).split('_')[0]}.tar"
        with fs.open(file, "r") as fr:
            size = simdjson.load(fr)["successes"]
        curr_sizes[tar_name] = size

    attempt = 0
    success = False
    size_json = curr_sizes
    while (not success) and (attempt <= 10):
        try:
            with fs.open(f"{url}/sizes.json", "w") as fo:
                simdjson.dump(size_json, fo)
            success = True
        except Exception as e:
            print(f"Exception as {e}")
            attempt += 1
            time.sleep(exponential_backoff(attempt))


def get_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Path containing monthly/yearly splits (s3 or local). "
        "Supported hierarchy path/split or path/split/shard.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=96, help="Number of processes to spawn."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    split_urls = list_timesplit_paths(args.url)

    multiprocessing.set_start_method("spawn")
    with Pool(args.num_proc) as p:
        res = list(tqdm(p.imap(generate_json, split_urls), total=len(split_urls)))
