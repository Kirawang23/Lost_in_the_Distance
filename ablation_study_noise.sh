
positions=("qa" "qna" "qnna" "qnnna")
noise_types=("novel" "random" "fineweb" "redpajama")
reverse_flags=(True False)
models=("gemini-1.5-flash")
task_names=("revname")
n_shots=(0)

for position in "${positions[@]}"; do
    for noise_type in "${noise_types[@]}"; do
        for reverse_flag in "${reverse_flags[@]}"; do
            for model in "${models[@]}"; do
                for task_name in "${task_names[@]}"; do
                    for n_shot in "${n_shots[@]}"; do
                        echo "Running script with position: $position, model: $model, reverse: $reverse_flag, task_name: $task_name, noise_type: $noise_type, n_shot: $n_shot"
                        python3 main.py --position "$position" --model "$model" --reverse "$reverse_flag" --task_name "$task_name" --noise_type "$noise_type" --n_shot "$n_shot"
                        if [ $? -ne 0 ]; then
                            echo "The script failed for position $position, model $model, reverse_flag $reverse_flag, task_name $task_name, noise_type $noise_type, n_shot $n_shot, retrying in 1 minute..."
                            sleep 60
                        else
                            echo "The script completed successfully for position $position, model $model, reverse_flag $reverse_flag, task_name $task_name, noise_type $noise_type, n_shot $n_shot."
                            break
                        fi
                    done
                done
            done
        done
    done
done

echo "All positions and reverse flags processed."
