# TiC-CLIP: Continual Training of CLIP Models

[[Paper]](https://arxiv.org/abs/2310.16226) [[OpenReview]](https://openreview.net/forum?id=TLADT8Wrhn) [[Tweet]](https://twitter.com/saurabh_garg67/status/1717571699720483023)

This repository contains training/evaluation code and data for the paper: 

**[TiC-CLIP: Continual Training of CLIP Models](https://arxiv.org/abs/2310.16226), Garg, S., Farajtabar, M., Pouransari, H., Vemulapalli, R., Mehta, S., Tuzel, O., Shankar, V. and Faghri, F., International Conference on Learning Representations (ICLR), 2024.**

- **Update 2024/06/06:** TiC-DataComp data and TiC-CILP models are now available on HuggingFace in [TiC-CLIP Collection](https://huggingface.co/collections/apple/tic-clip-666097407ed2edff959276e0).
- **Update 2024/03/19:** TiC-CLIP training/evaluation compatible with OpenCLIP/DataComp is released.
- **Update 2024/03/19:** TiC-CLIP camera ready copy is now available on [Arxiv](https://arxiv.org/abs/2310.16226) and [OpenReview](https://openreview.net/forum?id=TLADT8Wrhn).
- **Update 2024/01/16:** TiC-CLIP is accepted for poster presentation at [ICLR2024](https://iclr.cc), Vienna, Austria, May 7-11 2024.

## Overview


Our code is adapted from the 
[DataComp](https://github.com/mlfoundations/datacomp) code and we use 
[OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip) 
for continual training.   

We provide instructions and scripts to prepare training/evaluation data for 
TiC-DataComp, TiC-YFCC15m, and TiC-RedCaps. For all datasets, we require access 
to a copy of the original data, either locally or on a cloud storage (e.g., AWS 
S3). Please follow the original instructions for each dataset to acquire a copy 
of the dataset.

Experimental protocol:
![Experimental Protocol](files/images/exp_protocol.png)

Examples from our evaluation datasets: 
![Evaluation Dataset](files/images/examples.png)


## Installing dependencies

We use OpenCLIP and DataComp for training and evaluation. We have made minor modifications 
to OpenCLIP/DataComp for support of TiC evaluations and training. To checkout 
the specific version of each library and apply our corresponding patch run the 
following commands in order:
```bash
# Clone TiC-CLIP repository
git clone git@github.com:apple/ml-tic-clip.git
cd ml-tic-clip/

# Clone DataComp repository and apply patch
git clone https://github.com/mlfoundations/datacomp.git
cd datacomp
git checkout 7fdb5c653e70d9c6fcc63ac7c8c66843e7c6f3e8
git apply ../datacomp.patch  # Minor changes to support TiC training/evaluation
bash create_env.sh
cd ..

# Clone OpenCILP repository, apply patch, and install
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
git checkout 73fa7f03a33da53653f61841eb6d69aef161e521
git apply ../open_clip.patch  # Support for sampling without replacement
pip install --ignore-installed .
cd ..
```

To activate the environment:
```bash
conda activate datacomp
```

If using cloud storage services (e.g., AWS S3), you'll need to install 
additional dependencies (e.g. `pip install 'cloudpathlib[s3]'`).

## TiC-DataComp


### Download Original DataComp data

To obtain the training sets, we first need to download the commonpool data.
To download, run the following command, replacing `$scale` with the competition scale (i.e. `medium`, `large` or `xlarge`) and `$data_dir` with the output directory where you want the data to be stored.
```bash
cd datacomp
python download_upstream.py --scale $scale --data_dir $data_dir
```

Please refer to [DataComp](https://github.com/mlfoundations/datacomp/tree/main) for additional instructions.

Below, we provide UIDs files that can be used together with our patched copy `sharder.py` from DataComp to create TiC-DataComp datasets.

### TiC-DataComp UIDs

We provide the mapping of UIDs to yearly/monthly timesplits for the commonpool data (DataComp-`xlarge`).

- [TiC-DataComp (yearly data; no filter; no eval)] To download year -> UIDs for the non-filtered DataComp data after removal of all our evaluation UIDs. Number of samples per year are available as 
[sizes.json](https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_yearly_noeval/sizes.json).
```bash
for year in {2014..2023}; do
  wget https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_yearly_noeval/$year.npy
done
```
Although we do not provide specific UIDs files for basic filtering and bestpool filtering, it is straightforward to create the corresponding timesplits using the same UIDs files above to reshard a copy of basic filtered data or bestpool data.

- [TiC-DataComp (monthly data; no filter; including eval)]
To download month -> UIDs for the non-filtered DataComp data including all our evaluation UIDs. For ease of downloading, we grouped monthly UIDs files into yearly `.tar` archives. Number of samples per year are available as 
[sizes.json](https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_monthly/sizes.json).
```bash
for year in {2014..2023}; do
  wget https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_monthly/$year.tar
done
```
Again we do not provide specific UIDs files for filtered sets or excluding the evaluations as we can create the subsets by joining with other UIDs files.

- [TiC-DataComp-Retrieval Evaluation] UIDs: `https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_retrieval_evals_year2uids.pkl`.  This file contains a dictionary of year -> UIDs.
- [TiC-DataCompNet Evaluation (Classification)] UIDs: `https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacompnet_year2uids.pkl`. This file contains a nested dictionary of year -> synset -> UIDs.

### Training scripts

We provide scripts for reproducing our results for TiC-DataComp:
```
bash scripts/run_medium_basic_cumulative.sh
bash scripts/run_medium_basic_oracle.sh
bash scripts/run_medium_basic_sequential.sh
```

### Hyper-parameters

We fix hyperparameters used for training to original DataComp hyperparamets. For all scales, we train a ViT-B/16 model. The number of samples seen during training is determined by the scale, and is equal to the size of the corresponding pool we provide. Additional details on hyper-parameters can be found in the paper.

You should not modify any hyper-parameters for training, including batch size. Any changes may affect accuracy and make results incomparable.

### Example of preparing data, training, and evaluation

We provide a series of example commands for training and evaluating on TiC-DataComp-Medium.

Start by downloading the original DataComp data and setting the following variables:
```bash
DATACOMP_PATH=<s3 or local path to DataComp data>  # No trailing Slash
TIC_DATACOMP_PATH=<s3 or local path to store TiC-DataComp data>  # No trailing Slash
IMAGENET_VAL_PATH=<local path to imagenet validation data>
TIC_DATACOMP_Y_PATH="$TIC_DATACOMP_PATH/yearly"  # No trailing Slash
TIC_DATACOMP_M_PATH="$TIC_DATACOMP_PATH/monthly"  # No trailing Slash
DATA_Y_PATH="data/tic/datacomp/yearly"
DATA_M_PATH="data/tic/datacomp/monthly"
YEAR=2014
YEARMONTH=201403
mkdir -p $DATA_Y_PATH
```

Prepare the training data for TiC-DataComp-medium by resharding the DataComp-medium data:
```bash
## Download example ids for year $YEAR
wget -nc https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_yearly_noeval/$YEAR.npy -P $DATA_Y_PATH/train/

## Reshard data to create the subset for year $YEAR
python datacomp/resharder.py -i $DATACOMP_PATH -o $TIC_DATACOMP_Y_PATH/train/$YEAR/ -s $DATA_Y_PATH/train/$YEAR.npy

## Generate YEAR/sizes.json for each split. NOTE: do not pass YEAR in this path
python dataset_creation/tic-datacomp/add_sizes.py --url $TIC_DATACOMP_Y_PATH/train/ --num-proc 10

## Generate missing _stats.json files. NOTE: do not pass YEAR path
python dataset_creation/tic-datacomp/generate_stats.py --url $TIC_DATACOMP_Y_PATH/train/ --num-proc 10
```

We can now train using the DataComp code:
```bash
## Train Cummulative
pushd datacomp
final_data_dir=$TIC_DATACOMP_Y_PATH/train/$YEAR/
torchrun --nproc_per_node 8 --nnodes 1 \
    train.py \
    --scale "tic_medium" \
    --dataset_resampled \
    --data_dir $final_data_dir \
    --output_dir "./results/" \
    --exp_name "datacomp_medium-basic_cumulative" \
    --imagenet_val  $IMAGENET_VAL_PATH  \
    --save_frequency 1
popd

## Train Oracle
pushd datacomp
torchrun --nproc_per_node 8 --nnodes 1 \
    train.py \
    --scale "tic_medium" \
    --dataset_resampled \
    --data_dir $final_data_dir \
    --output_dir "./results/" \
    --exp_name "datacomp_medium-basic_cumulative" \
    --imagenet_val $IMAGENET_VAL_PATH  \
    --save_frequency 1 \
    --train_num_samples $((4096*4500*7))
popd
```

Next we prepare the retrieval evaluation data and evaluate the OpenAI ViT-B/16 model. Warning: This example prepares the evaluation for DataComp-medium scale but the final results are recommended to be evaluated on the xlarge scale.
```bash
# Generate numpy files with retrieval/datacompnet uids
python dataset_creation/tic-datacomp/generate_eval_uids.py --output-path $DATA_Y_PATH

# Create a webdataset with all evaluation data
# Next line takes ~30mins
python datacomp/resharder.py -i $DATACOMP_PATH -o $TIC_DATACOMP_Y_PATH/eval/all/ -s $DATA_Y_PATH/eval/alluids.npy

# Create a webdataset with only one year of data for each task
# Next line takes ~10s for tic-datacomp-medium (datacomp-128M)
python datacomp/resharder.py -i $TIC_DATACOMP_Y_PATH/eval/all -o $TIC_DATACOMP_Y_PATH/eval/retrieval/$YEAR/ -s $DATA_Y_PATH/eval/retrieval/$YEAR.npy

pushd datacomp
# Copy eval data locally
aws s3 cp $TIC_DATACOMP_Y_PATH/eval/retrieval/$YEAR/ $DATA_Y_PATH/eval/retrieval/$YEAR/ --recursive

# Add metadata to make compatible with build_wds_dataset function of clip-benchmark
bash ../dataset_creation/tic-datacomp/convert_wds.sh $DATA_Y_PATH/eval/retrieval/$YEAR/ retrieval

# Evaluate the OpenAI ViT-B/16 model on TiC/Retrieval/Yearly/2014
# WARNING: This evaluation is on the limited datacomp-medium subset for
# demonstration only. The final evaluation should be done on the datacomp-xlarge evalset
python ../dataset_creation/tic-datacomp/generate_tasklist.py --yaml-path tasklist.yml --sample-eval --eval-tasks retrieval/yearly
python evaluate.py --data_dir data/ --train_output_dir ./results --use_model "ViT-B-16 openai" --skip_hf --skip_db --skip_notification
popd
```

Next we prepare the DataCompNet evaluation data and evaluate the OpenAI ViT-B/16 model:
```bash
# Create the mapping of UIDs -> Classids for each year
python dataset_creation/tic-datacomp/generate_datacompnet_uid2classid.py --output-path $DATA_Y_PATH/eval/datacompnet

# Next line takes ~10s for tic-datacomp-medium (datacomp-128M) for one year
python datacomp/resharder.py -i $TIC_DATACOMP_Y_PATH/eval/all -o $TIC_DATACOMP_Y_PATH/eval/datacompnet/$YEAR/ -s $DATA_Y_PATH/eval/datacompnet/$YEAR.npy -c $DATA_Y_PATH/eval/datacompnet/uid2classid/$YEAR.pkl

pushd datacomp
# Copy eval data locally
aws s3 cp $TIC_DATACOMP_Y_PATH/eval/datacompnet/$YEAR/ $DATA_Y_PATH/eval/datacompnet/$YEAR/ --recursive

# Add metadata to make compatible with build_wds_dataset function of clip-benchmark
bash ../dataset_creation/tic-datacomp/convert_wds.sh $DATA_Y_PATH/eval/datacompnet/$YEAR/ classification

popd

# Evaluate the OpenAI ViT-B/16 model on TiC/Retrieval/Yearly/2014 and
# TiC/DataCompNet/Yearly/2014
# WARNING: This evaluation is on the limited datacomp-medium subset for
# demonstration only. The final evaluation should be done on the datacomp-xlarge evalset
python ../dataset_creation/tic-datacomp/generate_tasklist.py --yaml-path tasklist.yml --sample-eval --eval-tasks retrieval/yearly,datacompnet/yearly
python evaluate.py --data_dir data/ --train_output_dir ./results --use_model "ViT-B-16 openai" --skip_hf --skip_db --skip_notification
```

[Optional] In the paper we present restricted zero-shot accuracy for groups of ImageNet classes. The following script can be used to create the groups of synset IDs for our evaluations:
```bash
# Optional: calculated restricted accuracy on ImageNet
python dataset_creation/tic-datacomp/imagenet_restricted_classes.py --output-path $DATA_Y_PATH/eval/datacompnet
```

Next we create training data for the monthly splits. We use the yearly data to create and speed up part of this process.
```bash
# DatatComp-medium Monthly
mkdir -p $DATA_M_PATH
# Download monthly uids (including eval uids)
wget -nc https://docs-assets.developer.apple.com/ml-research/datasets/tic-clip/tic-datacomp_training_monthly/$YEAR.tar -P $DATA_M_PATH/traineval/
tar -xf $DATA_M_PATH/traineval/$YEAR.tar -C $DATA_M_PATH/traineval/

# **Make sure yearly data has been created first.** This will speedup the
# sharding
## Reshard data to create the subset for year $YEAR
python datacomp/resharder.py -i $TIC_DATACOMP_Y_PATH/train/$YEAR/ -o $TIC_DATACOMP_M_PATH/train/$YEARMONTH/ -s $DATA_M_PATH/traineval/$YEAR/$YEARMONTH.npy

## Generate YEAR/sizes.json for each split. NOTE: do not pass YEAR/YEARMONTH path
python dataset_creation/tic-datacomp/add_sizes.py --url $TIC_DATACOMP_M_PATH/train/ --num-proc 10

## Generate missing _stats.json files. NOTE: do not pass YEAR/YEARMONTH path
python dataset_creation/tic-datacomp/generate_stats.py --url $TIC_DATACOMP_M_PATH/train/ --num-proc 10
```

We can now train on the monthly data:
```bash
## Train Cummulative
pushd datacomp
final_data_dir=$TIC_DATACOMP_M_PATH/train/$YEARMONTH/
torchrun --nproc_per_node 8 --nnodes 1 \
    train.py \
    --scale "tic_medium" \
    --dataset_resampled \
    --data_dir $final_data_dir \
    --output_dir "./results/" \
    --exp_name "datacomp_medium-basic_cumulative" \
    --imagenet_val  $IMAGENET_VAL_PATH  \
    --save_frequency 1
popd

## Train Oracle
pushd datacomp
torchrun --nproc_per_node 8 --nnodes 1 \
    train.py \
    --scale "tic_medium" \
    --dataset_resampled \
    --data_dir $final_data_dir \
    --output_dir "./results/" \
    --exp_name "datacomp_medium-basic_cumulative" \
    --imagenet_val $IMAGENET_VAL_PATH  \
    --save_frequency 1 \
    --train_num_samples $((4096*4500*7))
popd
```

Next we create evaluation data for monthly splits and evaluate the OpenAI ViT-B/16 model.
```bash
# Create monthly eval data from the intersection of yearly eval data and the monthly train/eval uids
python datacomp/resharder.py -i $TIC_DATACOMP_Y_PATH/eval/retrieval/$YEAR/ -o $TIC_DATACOMP_M_PATH/eval/retrieval/$YEARMONTH/ -s $DATA_M_PATH/traineval/$YEAR/$YEARMONTH.npy
python datacomp/resharder.py -i $TIC_DATACOMP_Y_PATH/eval/datacompnet/$YEAR/ -o $TIC_DATACOMP_M_PATH/eval/datacompnet/$YEARMONTH/ -s $DATA_M_PATH/traineval/$YEAR/$YEARMONTH.npy -c $DATA_Y_PATH/eval/datacompnet/uid2classid/$YEAR.pkl

# Copy eval data locally
pushd datacomp
aws s3 cp $TIC_DATACOMP_M_PATH/eval/retrieval/$YEARMONTH/ $DATA_M_PATH/eval/retrieval/$YEARMONTH/ --recursive
aws s3 cp $TIC_DATACOMP_M_PATH/eval/datacompnet/$YEARMONTH/ $DATA_M_PATH/eval/datacompnet/$YEARMONTH/ --recursive

# Add metadata to make compatible with build_wds_dataset function of clip-benchmark
bash ../dataset_creation/tic-datacomp/convert_wds.sh $DATA_M_PATH/eval/retrieval/$YEARMONTH/ retrieval
bash ../dataset_creation/tic-datacomp/convert_wds.sh $DATA_M_PATH/eval/datacompnet/$YEARMONTH/ classification

# Evaluate the OpenAI ViT-B/16 model on TiC/DataComp/Retrieval/Monthly/2014 and
# TiC/DataCompNet/Monthly/201403
# WARNING: This evaluation is on the limited datacomp-medium subset for
# demonstration only. The final evaluation should be done on the datacomp-xlarge evalset
python ../dataset_creation/tic-datacomp/generate_tasklist.py --yaml-path tasklist.yml --sample-eval --eval-tasks retrieval/monthly,datacompnet/monthly
python evaluate.py --data_dir data/ --train_output_dir ./results --use_model "ViT-B-16 openai" --skip_hf --skip_db --skip_notification
```

## TiC-YFCC15M

Please follow these steps to setup the data:

- Please refer to these [instructions](https://gitlab.com/jfolz/yfcc100m/-/issues/2) to download the 15M subset of YFCC100M.
- Run `dataset_creation/tic-yfcc15m/create_splits.py` to create our year-group splits `/splits/{train,test}_batch_{batch}.csv` from `yfcc15m.csv`.
- Run `dataset_creation/create_tars.py` to create to convert each CSV and the original YFCC15m images to WebDataset format.
- One needs to add additional metadata for compatibility with DataComp data loaders. Our current codebase does not include special code to load this data but one can repurpose our scripts for TiC-DataComp.

## TiC-RedCaps

Please follow these steps to setup the data:

- Download and setup RedCaps following [original instructions](https://github.com/redcaps-dataset/redcaps-downloader).
- Run `dataset_creation/tic-redcaps/create_splits.py` to create our year-group splits `/splits/{train,test}_batch_{batch}.csv`.
- Run `dataset_creation/create_tars.py` to create to convert each CSV and the original RedCaps images to WebDataset format.
- One needs to add additional metadata for compatibility with DataComp data loaders. Our current codebase does not include special code to load this data but one can repurpose our scripts for TiC-DataComp.


## Citation

If you find this repository useful or use this code in your research, please cite the following paper: 

> Garg, S., Farajtabar, M., Pouransari, H., Vemulapalli, R., Mehta, S., Tuzel, O., Shankar, V. and Faghri, F., 2024. TiC-CLIP: Continual Training of CLIP Models. ICLR. 
```
@inproceedings{garg2024tic,
  title={TiC-CLIP: Continual Training of CLIP Models},
  author={Garg, Saurabh and Farajtabar, Mehrdad and Pouransari, Hadi and Vemulapalli, Raviteja and Mehta, Sachin and Tuzel, Oncel and Shankar, Vaishaal and Faghri, Fartash},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://openreview.net/forum?id=TLADT8Wrhn}
}
```
