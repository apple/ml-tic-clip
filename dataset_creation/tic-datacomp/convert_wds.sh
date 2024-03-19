#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
#!/bin/bash
# Convert a webdataset created by DataComp/resharder.py to one compatible with
# build_dataset in clip-benchmark
[ $# -lt 2 ] && echo "Arg1: path, Arg2: retrieval/classification." && exit
pushd $1
# Remove leading 0s from shards
for i in [0-9]*.tar; do j=${i%.*}; mv $i $((10#$j)).tar; mv ${i%.tar}_stats.json $((10#$j)).json; done
# Save number of shards in nshards.txt
sed -rn 's/^.*"output_shard_count": ([[:digit:]]+),.*$/\1/p' meta.json > nshards.txt
# Save number of shards in count.txt
sed -rn 's/^.*"output_count": ([[:digit:]]+),.*$/\1/p' meta.json > count.txt
echo $2 > dataset_type.txt
if [ $2 == "classification" ]
then
    wget -nc "https://huggingface.co/datasets/djghosh/wds_imagenet1k_test/raw/main/zeroshot_classification_templates.txt"
    wget -nc "https://huggingface.co/datasets/djghosh/wds_imagenet1k_test/raw/main/classnames.txt"
fi
# Test that the dataset can be loaded in python using clip-benchmark
# python -c 'from clip_benchmark.datasets.builder import build_dataset; ds=build_dataset(dataset_name="wds/tic/datacomp/retrieval/$YEAR", transform=None, root="./", split="", task="$2"); print(ds)'
popd
