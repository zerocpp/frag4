#!/bin/bash
# Description: Script to run the evaluation of the model on the generated responses

models=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
datasets=("squad" "triviaqa")
samples=("greedy_golden" "greedy_without" "greedy_irrelevant")
splits=("train" "test" "validation")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for sample in "${samples[@]}"; do
            for split in "${splits[@]}"; do
                echo "$split"
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

