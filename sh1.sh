# Description: Generate greedy responses on SQuAD-test
cuda=1
models=("meta-llama/Llama-3.1-8B-Instruct")
datasets=("squad" "triviaqa")
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        DEVICE=cuda:${cuda} python generate_responses.py --dataset "${dataset}" --split test --model "${model}" --num_samples 1000 --num_generations 1 --temperature 0.1 --use_context --irrelevant_context --return_latent
        DEVICE=cuda:${cuda} python generate_responses.py --dataset "${dataset}" --split test --model "${model}" --num_samples 1000 --num_generations 1 --temperature 0.1 --use_context --no-irrelevant_context --return_latent
        DEVICE=cuda:${cuda} python generate_responses.py --dataset "${dataset}" --split test --model "${model}" --num_samples 1000 --num_generations 1 --temperature 0.1 --no-use_context --irrelevant_context --return_latent
    done
done
