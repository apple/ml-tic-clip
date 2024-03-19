#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""Create train/test .csv files for each year-group from yfcc15m.csv."""
import re
import ftfy
import urllib
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def remove_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


def clean_text(x):
    return remove_html(ftfy.fix_text(urllib.parse.unquote_plus(x)))


years = []
years[0] = ["2004", "2005", "2006", "2007", "2008"]
years[1] = ["2009", "2010"]
years[2] = ["2011", "2012"]
years[3] = ["2013", "2014"]


reverse_dates = {}
for i in years:
    for time in years[i]:
        reverse_dates[time] = i

df = pd.read_csv("./yfcc15m.csv")

split_dfs = []
for i in range(4):
    split_dfs[i] = []

count = 0
for idx, row in tqdm(df.iterrows()):
    date = str(datetime.datetime.fromtimestamp(row["dateuploaded"]).year)
    if pd.isna(row["title"]) or pd.isnull(row["title"]):
        row["title"] = ""
    if pd.isna(row["description"]) or pd.isnull(row["description"]):
        row["description"] = ""
    if pd.isna(row["usertags"]) or pd.isnull(row["usertags"]):
        row["usertags"] = ""
    caption = clean_text(f"{row['title']}. {row['description']}. {row['usertags']}")[
        :500
    ]
    img_path = f"{row['key'][:3]}/{row['key'][3:6]}/{row['key']}.jpg"
    if date not in reverse_dates:
        count += 1
        continue
    split_dfs[reverse_dates[date]].append({"path": img_path, "caption": caption})


print(f"{count} not mapped paths")


for batch in range(4):
    split_dfs[batch] = pd.DataFrame.from_records(split_dfs[batch])
    train, test = train_test_split(split_dfs[batch], test_size=0.01)

    train.to_csv(f"./splits/train_batch_{batch}.csv", sep="\t", index=False)
    test.to_csv(f"./splits/test_batch_{batch}.csv", sep="\t", index=False)
