#!/bin/bash
# GPU-0执行的脚本

#################不同的参数##################
cuda=0
models=("Qwen/Qwen2.5-7B-Instruct")
# models=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")


# 记录开始时间
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"


#################generate##################

context_types=("irrelevant" "golden" "without")
contexts=("--use_context --irrelevant_context" "--use_context --no-irrelevant_context" "--no-use_context --irrelevant_context")

sample_prefixs=("greedy" "sample")
num_generations=(1 30)
temperatures=(0.1 1.0)
latents=("--return_latent" "--no-return_latent")

datasets=("squad" "triviaqa" "bioasq")
splits=("train" "test" "validation")
num_samples=(2000 100 100)
# num_samples=(10000 1000 1000)

gen_override="--no-override"
cluster_override="--no-override"
root_dir="."

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*3*3*3*2))

for model in "${models[@]}"; do # 1
    for dataset in "${datasets[@]}"; do # 3
        for k in "${!splits[@]}"; do # 3
            split="${splits[$k]}"
            num_sample="${num_samples[$k]}"
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
                    dataset_json_file="$root_dir/dataset/json/${dataset}_${split}_${num_sample}.json"
                    output_dir="$root_dir/output/${split}/generation/${model}/${dataset}/${sample}"
                    cmd="DEVICE=cuda:$cuda python generate_responses.py --output_dir $output_dir --model $model --num_generations $num_generation --temperature $temperature $context $latent --dataset_json_file $dataset_json_file $gen_override"
                    echo "> $cmd"
                    eval $cmd
                done
            done
        done
    done
done


#################cluster##################

# 任务计数器
task_counter=0
# 任务总数
task_total=$((1*3*3*3))

for model in "${models[@]}"; do # 1
    for dataset in "${datasets[@]}"; do # 3
        for k in "${!splits[@]}"; do # 3
            split="${splits[$k]}"
            num_sample="${num_samples[$k]}"
            for sample_suffix in "${context_types[@]}"; do # 3
                task_counter=$((task_counter + 1))
                echo "Task2: $task_counter/$task_total"
                sample="sample_${sample_suffix}"
                dataset_json_file="$root_dir/dataset/json/${dataset}_${split}_${num_sample}.json"
                input_dir="$root_dir/output/${split}/generation/${model}/${dataset}/${sample}"
                output_dir="$root_dir/output/${split}/clustered/${model}/${dataset}/${sample}"
                cmd="DEVICE=cuda:$cuda python cluster_responses.py --dataset_json_file $dataset_json_file --input_dir $input_dir --output_dir $output_dir $cluster_override"
                echo "> $cmd"
                eval $cmd
            done
        done
    done
done


# 记录结束时间
end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"
echo "End time: $end_time"

# 计算时间差
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
echo "Time cost: $((end_seconds-start_seconds))s"
# 按时分秒显示时间差
echo "Time cost: $(date -d "1970-01-01 $((end_seconds-start_seconds)) sec" +"%H:%M:%S")"
