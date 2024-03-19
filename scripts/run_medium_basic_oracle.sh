#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
pushd datacomp
pip install -r requirements.txt


year=2022
num_gpus=8
scale="tic_medium"
filtering="basic"
output_dir="<path to output dir>" #  s3 path or local dir
exp_name="datacomp_${scale}-${filtering}_oracle"
train_num_samples=$((4096*4500*7))  # 7x more iterations for the Oracle
data_dir="<path to output dir>" # s3 path or local dir for $filtering technique
imagenet_validation_data="local/imagenet/path"
final_data_dir="${data_dir}/2014/"

for y in {2015..year}; do 
    final_data_dir=$data_dir"/${y}/::"$final_data_dir
done

torchrun --nproc_per_node 8 --nnodes 1 \
--max_restarts=10000 \
--rdzv_backend c10d \
--rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
--rdzv_conf "timeout=3000,read_timeout=10000" \
train.py  --scale $scale --dataset_resampled --train_num_samples $train_num_samples --data_dir $final_data_dir --output_dir $output_dir --exp_name $exp_name --imagenet_val  $imagenet_validation_data  --save_frequency 1  --report_to_wandb --wandb_project_name Tic-datacomp-${scale}-${filtering}-oracle

sleep 3600
