#!/bin/bash
# 存放待执行的脚本

################# cuda0.sh #################
cuda=0

###########################################
# Description: Script to generate responses
models=("Qwen/Qwen2.5-7B-Instruct")
# models=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")

context_types=("irrelevant" "golden" "without")
contexts=("--use_context --irrelevant_context" "--use_context --no-irrelevant_context" "--no-use_context --irrelevant_context")

sample_prefixs=("greedy" "sample")
num_generations=(1 10)
temperatures=(0.1 1.0)
latents=("--return_latent" "--no-return_latent")

datasets=("squad" "triviaqa")
splits=("train" "test" "validation")

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*2*3*3*2))

for model in "${models[@]}"; do # 1
    for dataset in "${datasets[@]}"; do # 2
        for split in "${splits[@]}"; do # 3
            for i in "${!context_types[@]}"; do # 3
                context_type="${context_types[$i]}"
                context="${contexts[$i]}"
                for j in "${!sample_prefixs[@]}"; do # 2
                    task_counter=$((task_counter + 1))
                    echo "Task1: $task_counter/$task_total"
                    sample_prefix="${sample_prefixs[$j]}"
                    num_generation="${num_generations[$j]}"
                    temperature="${temperatures[$j]}"
                    latent="${latents[$j]}"
                    sample="${sample_prefix}_${context_type}"
                    dataset_json_file="output/dataset/${dataset}_${split}.json"
                    output_dir="output/${split}/generation/${model}/${dataset}/${sample}"
                    cmd="DEVICE=cuda:$cuda python generate_responses.py --output_dir $output_dir --model $model --num_generations $num_generation --temperature $temperature $context $latent --dataset_json_file $dataset_json_file --no-override"
                    echo "> $cmd"
                    eval $cmd
                done
            done
        done
    done
done


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








################# cuda1.sh #################
cuda=1

###########################################
# Description: Script to generate responses
models=("meta-llama/Llama-3.1-8B-Instruct")
# models=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")

context_types=("irrelevant" "golden" "without")
contexts=("--use_context --irrelevant_context" "--use_context --no-irrelevant_context" "--no-use_context --irrelevant_context")

sample_prefixs=("greedy" "sample")
num_generations=(1 10)
temperatures=(0.1 1.0)
latents=("--return_latent" "--no-return_latent")

datasets=("squad" "triviaqa")
splits=("train" "test" "validation")

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*2*3*3*2))

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for split in "${splits[@]}"; do
            for i in "${!context_types[@]}"; do
                context_type="${context_types[$i]}"
                context="${contexts[$i]}"
                for j in "${!sample_prefixs[@]}"; do
                    task_counter=$((task_counter + 1))
                    echo "Task1: $task_counter/$task_total"
                    sample_prefix="${sample_prefixs[$j]}"
                    num_generation="${num_generations[$j]}"
                    temperature="${temperatures[$j]}"
                    latent="${latents[$j]}"
                    sample="${sample_prefix}_${context_type}"
                    dataset_json_file="output/dataset/${dataset}_${split}.json"
                    output_dir="output/${split}/generation/${model}/${dataset}/${sample}"
                    cmd="DEVICE=cuda:$cuda python generate_responses.py --output_dir $output_dir --model $model --num_generations $num_generation --temperature $temperature $context $latent --dataset_json_file $dataset_json_file --no-override"
                    echo "> $cmd"
                    eval $cmd
                done
            done
        done
    done
done


###########################################
# Description: Script to cluster responses
models=("meta-llama/Llama-3.1-8B-Instruct")
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


################# cuda.sh #################
# Description: 评估贪婪生成的结果

# models=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
models=("Qwen/Qwen2.5-7B-Instruct")
datasets=("squad" "triviaqa")
samples=("greedy_golden" "greedy_without" "greedy_irrelevant")
splits=("train" "test" "validation")

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*2*3*3))

for model in "${models[@]}"; do # 1
    for dataset in "${datasets[@]}"; do # 2
        for sample in "${samples[@]}"; do # 3
            for split in "${splits[@]}"; do # 3
                task_counter=$((task_counter + 1))
                echo "Task2: $task_counter/$task_total"
                dataset_json_file="output/dataset/${dataset}_${split}.json"
                input_dir="output/${split}/generation/${model}/${dataset}/${sample}"
                output_dir="output/${split}/evaluation/${model}/${dataset}/${sample}"
                cmd="python evaluate_responses.py --input_dir $input_dir --output_dir $output_dir --dataset_json_file $dataset_json_file --no-override"
                echo "> $cmd"
                eval $cmd
            done
        done
    done
done
