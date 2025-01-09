#!/bin/bash

#################记录开始时间##################
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

#################不同的参数##################
cuda=1
model="Qwen/Qwen2.5-7B-Instruct"
num_generations=10
dataset_names=("hotpotqa" "msmarco" "nfcorpus" "scidocs" "scifact" "nq")
# dataset_size="toy"
dataset_size="all"
# dataset_size="large"
# dataset_size="small100"
# dataset_jsonl_path="dataset/rank/nq-rank-10.jsonl"
# output_dir="output/rank/gen/Qwen/Qwen2.5-7B-Instruct/nq-rank-10"
gen_override="--no-override"

#################generate##################
for dataset_name in ${dataset_names[@]}; do
    index=$(($index + 1))
    total=${#dataset_names[@]}
    echo "Processing dataset $index of $total: $dataset_name"
    dataset_jsonl_path="dataset/rank/${dataset_name}/${dataset_name}-${dataset_size}.jsonl"
    output_dir="output/rank/gen/Qwen/Qwen2.5-7B-Instruct/${dataset_name}"
    cmd="DEVICE=cuda:$cuda python rank_gen.py --output_dir $output_dir --model $model --num_generations $num_generations --dataset_jsonl_path $dataset_jsonl_path $gen_override"
    echo "> $cmd"
    eval $cmd
done

#################time-log##################
# 记录结束时间
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"
echo "End time: $end_time"
# 计算时间差
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
echo "Time cost: $((end_seconds-start_seconds))s"
