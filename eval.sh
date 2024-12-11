
#################evaluate##################
# python evaluate_responses.py --eval_model_type ollama --eval_model_name qwen2.5:72b-instruct-q4_0 --temperature 0.01 --scores_key qwen_scores --input_dir output/train/generation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/train/evaluation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --dataset_json_file data/squad/sample_golden.json
models=("meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
datasets=("squad" "triviaqa")
splits=("train" "test" "validation")
# num_samples=(2000 100 100)
num_samples=(10000 1000 1000)
samples=("greedy_golden" "greedy_without" "greedy_irrelevant")
root_dir="."
eval_override="--override"

# 任务计数器
task_counter=0
# 任务总数
task_total=$((2*2*3*3))

starttime=`date +'%Y-%m-%d %H:%M:%S'`

for model in "${models[@]}"; do # 2
    for dataset in "${datasets[@]}"; do # 2
        for k in "${!splits[@]}"; do # 3
            split="${splits[$k]}"
            num_sample="${num_samples[$k]}"
            for sample in "${samples[@]}"; do # 3
                task_counter=$((task_counter + 1))
                echo "Eval: $task_counter/$task_total"
                dataset_json_file="$root_dir/output/dataset/${dataset}_${split}_${num_sample}.json"
                input_dir="$root_dir/output/${split}/generation/${model}/${dataset}/${sample}"
                output_dir="$root_dir/output/${split}/evaluation/${model}/${dataset}/${sample}"
                cmd="python evaluate_responses.py --scores_key qwen_scores --input_dir $input_dir --output_dir $output_dir --dataset_json_file $dataset_json_file $eval_override"
                echo "> $cmd"
                eval $cmd
            done
        done
    done
done

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "开始时间： "$starttime
echo "结束时间： "$endtime
echo "本次运行时间： "$((end_seconds-start_seconds))"s"  # 共计04:30:00
