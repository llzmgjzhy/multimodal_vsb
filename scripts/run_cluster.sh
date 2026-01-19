model=prototype_stats
batch_size=32
epochs=100
lr=1e-4
itr=1
task=cluster # fault_detection or classification

python run_cluster.py \
    --task $task \
    --comment "$task using $model" \
    --name "${task}_vsb" \
    --root_path ./dataset \
    --data_path VSBdata \
    --output_dir ./tensorboard \
    --records_file vsb_$task.xlsx \
    --model_name $model \
    --epochs $epochs \
    --loss cluster \
    --key_metric loss \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --d_model 64 \
    --dropout 0.1 \
    --patience 20 \
    --phase_level \

    # --weight_decay 1e-3 \
