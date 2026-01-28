model=classifier_set_transformer
batch_size=32
epochs=50
lr=1e-4
itr=1
task=fault_detection # fault_detection or classification

python run_stage2.py \
    --task $task \
    --comment "$task using $model" \
    --name "${task}_vsb" \
    --root_path ./dataset \
    --data_path VSBdata \
    --output_dir ./tensorboard \
    --records_file vsb_$task.xlsx \
    --model_name $model \
    --epochs $epochs \
    --loss bce \
    --key_metric loss \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --d_model 64 \
    --dropout 0.1 \
    --patience 20 \
    --phase_level \
    --n_layers 3 \
    --n_heads 4 \
    --d_ff 128 \
    --cluster_dir cluster_vsb_2026-01-28_17-58-32_tZU
    # --weight_decay 1e-3 \
