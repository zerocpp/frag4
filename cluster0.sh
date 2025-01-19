#!/bin/bash

#################记录开始时间##################
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# 以时间戳为文件名，创建一个日志文件
timestamp=$(date "+%Y%m%d-%H%M%S")
log_file="log/cluster0-$timestamp.log"
touch $log_file

#################不同的参数##################
cuda=0
model="Qwen/Qwen2.5-7B-Instruct"
dataset_names=("trec-covid" "climate-fever" "dbpedia-entity" "fever" "fiqa")
# dataset_size="toy"
# dataset_size="small"
# dataset_size="large"
# dataset_size="small100"
dataset_size="all"
# dataset_jsonl_path="dataset/rank/nq-rank-10.jsonl"
# output_dir="output/rank/gen/Qwen/Qwen2.5-7B-Instruct/nq-rank-10"
cluster_override="--no-override"

#################聚类cluster##################
for dataset_name in ${dataset_names[@]}; do
    index=$(($index + 1))
    total=${#dataset_names[@]}
    echo "Processing dataset $index of $total: $dataset_name"
    dataset_path="dataset/rank/${dataset_name}/${dataset_name}-${dataset_size}.jsonl"
    input_dir="output/rank/gen/Qwen/Qwen2.5-7B-Instruct/${dataset_name}"
    output_dir="output/rank/cluster/Qwen/Qwen2.5-7B-Instruct/${dataset_name}"
    cmd="DEVICE=cuda:$cuda python rank_cluster.py --input_dir $input_dir --output_dir $output_dir --dataset_path $dataset_path $cluster_override >> $log_file 2>&1"
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
