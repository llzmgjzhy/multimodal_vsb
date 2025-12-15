model=one_fits_all
batch_size=32
epochs=100
lr=1e-4
itr=1
task=classification

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
    --loss cross_entropy \
    --key_metric loss \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --d_model 768 \
    --dropout 0.21 \
    --patience 100 \
    # --weight_decay 1e-3 \
