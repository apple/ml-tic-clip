#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""Make TAR files containing images and annotations from a single CSV file.

These TAR files contain maximum `Z` instances each. For example, a file
named "input.csv" with 2500 instances and Z = 1000 would make three TAR files:

    {output_dir}/{input}_00000000.tar : instance ID [   0 to  999] from CSV
    {output_dir}/{input}_00000001.tar : instance ID [1000 to 1999] from CSV
    {output_dir}/{input}_00000002.tar : instance ID [2000 to 2499] from CSV
"""
import argparse
import json
import tarfile
import tempfile
from pathlib import Path
import pandas as pd
import os

# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", help="Path to CSV containing data path and caption.")
parser.add_argument(
    "--image-dir", help="""Path to directory containing RedCaps images. This
    directory could either have image files (jpg) or nested sub-directories
    per subreddit, each containing images.""",
)
parser.add_argument("--output-dir", help="Save generated TAR files here.")
parser.add_argument(
    "-z", "--shard-size", type=int, default=1000,
    help="Path prefix for saving TAR files.",
)
# fmt: on


def main(_A: argparse.Namespace):

    df = pd.read_csv(Path(_A.input), sep="\t")

    # Create output directory if it does not exist.
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / Path(_A.input).stem)

    IMAGE_DIR = Path(_A.image_dir)

    # 1. Keep track of the current index of TAR file shard and dataset index.
    # 2. Count number of images (and their annotations) added to TAR files.
    # 3. Count number of annotations skipped because their image was absent.
    SHARD_INDEX, ADDED_ANNS, SKIPD_ANNS = f"{0:0>8d}", 0, 0

    # Create TAR file handle for the initial shard.
    tar_handle = tarfile.open(f"{output_prefix}__{SHARD_INDEX}.tar", "w")

    for idx, row in df.iterrows():
        image_path = IMAGE_DIR / f"{row['path']}"

        # Skip current annotation if its image does not exist.
        if not image_path.exists():
            SKIPD_ANNS += 1
            continue

        # Dump annotation JSON to a temporary file to add in TAR.
        with tempfile.NamedTemporaryFile("w+") as tmpfile:
            add_in_tar = {"caption": row["caption"]}
            tmpfile.write(json.dumps(add_in_tar))
            tmpfile.seek(0)

            # Add image (JPG) and annotation (JSON) in TAR file.
            tar_handle.add(image_path, arcname=f"{idx}.jpg")
            tar_handle.add(tmpfile.name, arcname=f"{idx}.json")

        ADDED_ANNS += 1

        os.remove(image_path)

        # Close TAR file shard to finalize current shard once it is full
        # with `Z` instances. Then create a new shard.
        if ADDED_ANNS % _A.shard_size == 0 and ADDED_ANNS > 0:
            tar_handle.close()
            print(f"Saved shard: {output_prefix}__{SHARD_INDEX}.tar")

            SHARD_INDEX = f"{int(SHARD_INDEX) + 1:0>8d}"
            tar_handle = tarfile.open(f"{output_prefix}__{SHARD_INDEX}.tar", "w")

    # Close the last TAR file handle to properly save it.
    tar_handle.close()
    print(f"Saved shard: {output_prefix}__{SHARD_INDEX}.tar\n")
    print(f"Skipped {SKIPD_ANNS} annotations due to missing images.")


if __name__ == "__main__":
    _A = parser.parse_args()

    print("Command line args:")
    for arg in vars(_A):
        print(f"{arg:<20}: {getattr(_A, arg)}")

    main(_A)
