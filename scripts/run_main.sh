model=linear
batch_size=32
epochs=100
lr=1e-4
itr=1
task=fault_detection

python run_main.py \
    --task $task \
    --comment "$task using $model" \
    --name "${task}_vsb" \
    --root_path ./dataset \
    --data_path VSBdata \
    --output_dir ./tensorboard \
    --records_file vsb_$task.xlsx \
    --model_name $model \
    --epochs $epochs \
    --loss focal \
    --key_metric mcc \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --d_model 768 \
    --dropout 0.1 \
    --patience 20 \
    # --weight_decay 1e-3 \
