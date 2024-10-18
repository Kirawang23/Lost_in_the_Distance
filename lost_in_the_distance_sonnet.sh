positions=("qnnna" "qnna" "qna" "qa")
reverse_flags=(True False)
models=("claude-3-5-sonnet-20240620")
max_tokens_=(20000)
task_names=("revname")
noise_types=("novel")


for reverse_flag in "${reverse_flags[@]}"; do
    for position in "${positions[@]}"; do
        for model in "${models[@]}"; do
            for max_tokens in "${max_tokens_[@]}"; do
                for task_name in "${task_names[@]}"; do
                    for noise_type in "${noise_types[@]}"; do
                        echo "Running script with position: $position, model: $model, reverse: $reverse_flag, task_name: $task_name, noise_type: $noise_type, max_tokens: $max_tokens"
                        python3 main.py --position "$position" --model "$model" --reverse "$reverse_flag" --task_name "$task_name" --noise_type "$noise_type" --max_tokens "$max_tokens"
                        if [ $? -ne 0 ]; then
                            echo "The script failed for position $position, model $model, reverse_flag $reverse_flag, task_name $task_name, noise_type $noise_type, max_tokens $max_tokens. Retrying in 60 seconds."
                            sleep 60
                        else
                            echo "The script completed successfully for position $position, model $model, reverse_flag $reverse_flag, task_name $task_name, noise_type $noise_type, max_tokens $max_tokens."
                            break
                        fi
                    done
                done
            done
        done
    done
done

echo "All positions and reverse flags processed."
