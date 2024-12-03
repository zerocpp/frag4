#!/bin/bash
# GPU-1执行的脚本
cuda=1

###########################################
# Description: Script to cluster responses
models=("Qwen/Qwen2.5-7B-Instruct")
datasets=("squad" "triviaqa")
splits=("train" "test" "validation")
sample_suffixs=("irrelevant" "golden" "without")
override="--no-override"
root_dir="."

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*2*3*3))

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for split in "${splits[@]}"; do
            for sample_suffix in "${sample_suffixs[@]}"; do
                task_counter=$((task_counter + 1))
                echo "Task2: $task_counter/$task_total"
                sample="sample_${sample_suffix}"
                dataset_json_file="$root_dir/output/dataset/${dataset}_${split}.json"
                input_dir="$root_dir/output/${split}/generation/${model}/${dataset}/${sample}"
                output_dir="$root_dir/output/${split}/clustered/${model}/${dataset}/${sample}"
                cmd="DEVICE=cuda:$cuda python cluster_responses.py --dataset_json_file $dataset_json_file --input_dir $input_dir --output_dir $output_dir $override"
                echo "> $cmd"
                eval $cmd
            done
        done
    done
done
