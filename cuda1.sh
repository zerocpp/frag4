#!/bin/bash
# Description: Script to generate responses

cuda=1
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

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for split in "${splits[@]}"; do
            for i in "${!context_types[@]}"; do
                context_type="${context_types[$i]}"
                context="${contexts[$i]}"
                for j in "${!sample_prefixs[@]}"; do
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
