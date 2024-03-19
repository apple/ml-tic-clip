#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
import tarfile
import io
import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import simdjson
import time
from dataset_creation.utils import fsspec_url_to_fs, exponential_backoff, list_timesplit_paths


def process_stats(inp_file_path):
    """Generate xyz_stats.json file for the path to a xyz.tar file."""
    success = False
    attempt = 1
    while (not success) and (attempt <= 10):
        try:
            file_path = "s3://" + inp_file_path
            fs, url = fsspec_url_to_fs(file_path)
            with fs.open(url, "rb") as ff:
                bytes = io.BytesIO(ff.read())
                tar = tarfile.open(fileobj=bytes)
                count = len([n for n in tar.getnames() if "json" in n])
            out_path = "/".join(file_path.split("/")[:-1])
            base, ext = os.path.splitext(os.path.basename(file_path))
            out_file = os.path.join(out_path, base + "_stats.json")
            with fs.open(out_file, "w") as fo:
                simdjson.dump({"successes": count}, fo)
            success = True
        except Exception as e:
            print(f"Exception as {e}")
            attempt += 1
            time.sleep(exponential_backoff(attempt))
    return count


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
    remaining_files = []
    for url in split_urls:
        fs, url = fsspec_url_to_fs(url)
        files = fs.ls(url)
        done_files = set(
            os.path.splitext(os.path.basename(f))[0].split("_")[0]
            for f in files
            if "_stats.json" in f
        )
        all_files_tar = set(
            os.path.splitext(os.path.basename(f))[0].split(".")[0]
            for f in files
            if ".tar" in f
        )
        remaining_files.extend(f"{url}/{f}.tar" for f in all_files_tar - done_files)
    print(len(remaining_files))

    multiprocessing.set_start_method("spawn")
    with Pool(args.num_proc) as p:
        res = list(
            tqdm(
                p.imap(process_stats, remaining_files),
                total=len(remaining_files),
                mininterval=60,
            )
        )

    print(sum(res))
    print("done")
