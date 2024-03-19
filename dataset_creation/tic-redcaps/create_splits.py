#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""Create train/test .csv files for each year-group from .json files."""
import json
from typing import Any, Dict
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


annotations_folder = "./annotations/"

split_dfs = {}
for year in ["2017", "2018", "2019", "2020"]:
    print(f"Year: {year}")
    split_dfs[year] = []
    for annotations_filepath in tqdm(glob.glob(f"{annotations_folder}*_{year}.json")):
        ANNOTATIONS: Dict[str, Any] = json.load(open(annotations_filepath))

        for ann in ANNOTATIONS["annotations"]:
            img_path = f"{ann['subreddit']}/{ann['image_id']}.jpg"
            caption = ann["caption"]

            split_dfs[year].append({"path": img_path, "caption": caption})


for batch in ["2017", "2018", "2019", "2020"]:
    split_dfs[batch] = pd.DataFrame.from_records(split_dfs[batch])
    train, test = train_test_split(split_dfs[batch], test_size=0.01)

    train.to_csv(f"./splits/train_batch_{batch}.csv", sep="\t", index=False)
    test.to_csv(f"./splits/test_batch_{batch}.csv", sep="\t", index=False)
