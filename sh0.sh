# cluster squad llama8b
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_without --output_dir output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_without
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_irrelevant --output_dir output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_irrelevant --no-override

# bash eval.sh
# # eval.sh begin
# models=("meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
# datasets=("squad" "triviaqa")
# samples=("greedy_golden" "greedy_without" "greedy_irrelevant")

# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         for sample in "${samples[@]}"; do
#             echo "> start"
#             echo "> model: $model, dataset: $dataset, sample: $sample"
#             input_dir="output/${model}/${dataset}/${sample}"
#             output_dir="output/eval/${model}/${dataset}/${sample}"
#             echo "> input_dir: $input_dir"
#             echo "> output_dir: $output_dir"
#             python evaluate_responses.py --eval_model_type ollama --eval_model_name qwen2.5:72b-instruct-q4_0 --temperature 0.01 --scores_key qwen_scores --input_dir "$input_dir" --output_dir "$output_dir"
#             echo "> done"
#         done
#     done
# done
# # eval.sh end



# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --no-use_context --irrelevant_context
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --use_context --no-irrelevant_context
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --use_context --irrelevant_context
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --no-use_context --irrelevant_context
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --use_context --no-irrelevant_context --return_latent
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --use_context --irrelevant_context --return_latent
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset triviaqa --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --no-use_context --irrelevant_context --return_latent

# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --use_context --no-irrelevant_context

# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --use_context --irrelevant_context --return_latent
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --use_context --no-irrelevant_context --return_latent
# DEVICE=cuda:0 MAX_MEMORY_CONFIG='{0:"24GIB", 1:"0GIB"}' python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 1 --retry_times 3 --temperature 0.1 --max_new_tokens 50 --no-use_context --irrelevant_context --return_latent

# # cluster squad qwen7b
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/squad/sample_without --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/squad/sample_without
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/squad/sample_irrelevant --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/squad/sample_irrelevant

# # cluster triviaqa qwen7b
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset triviaqa --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_golden --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_golden
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset triviaqa --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_without --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_without
# DEVICE=cuda:0 python cluster_responses.py --model microsoft/deberta-v2-xxlarge-mnli --dataset triviaqa --split train --num_samples 2000 --input_dir output/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_irrelevant --output_dir output/clustered/Qwen/Qwen2.5-7B-Instruct/triviaqa/sample_irrelevant
