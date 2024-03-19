#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
pushd datacomp
pip install -r requirements.txt


num_gpus=8
scale="tic_medium"
filtering="basic"
output_dir="<path to output dir>" #  s3 path or local dir
exp_name="datacomp_xlarge-${filtering}_cumulative"
data_dir="<path to output dir>" # s3 path or local dir for $filtering technique
imagenet_validation_data="local/imagenet/path"

final_data_dir="${data_dir}/2014/"

for year in 2015 2016; do 
    final_data_dir=$data_dir"/${year}/::"$final_data_dir
done


############### For first run #####################


torchrun --nproc_per_node 8 --nnodes 1 \
--max_restarts=10000 \
--rdzv_backend c10d \
--rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
--rdzv_conf "timeout=3000,read_timeout=10000" \
train.py --scale $scale --dataset_resampled --data_dir $final_data_dir --output_dir $output_dir --exp_name $exp_name --imagenet_val  $imagenet_validation_data  --save_frequency 1  --report_to_wandb --wandb_project_name tic-datacomp-${scale}-${filtering}-cumulative-${year}



############### For subsequent runs ################

year=2022 # Choose any year from 2017 to 2022

for y in {2017..year}; do 
    final_data_dir=$data_dir"/${y}/::"$final_data_dir
done

checkpoint="<s3 or local path to prev timestamp checkpoint>" ## For the first run omit the checkpoint path and initialize randomly 

torchrun --nproc_per_node 8 --nnodes 1 \
--max_restarts=10000 \
--rdzv_backend c10d \
--rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
--rdzv_conf "timeout=3000,read_timeout=10000" \
train.py --warmup 0 --new_run --resume $checkpoint --scale $scale --dataset_resampled --data_dir $final_data_dir --output_dir $output_dir --exp_name $exp_name --imagenet_val  $imagenet_validation_data  --save_frequency 1  --report_to_wandb --wandb_project_name tic-datacomp-${scale}-${filtering}-cumulative-${year}

sleep 3600
