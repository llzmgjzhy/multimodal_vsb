model=swin_transformer
model_pretrain="microsoft/swin-tiny-patch4-window7-224"
batch_size=32
epochs=20
lr=1e-4
itr=1
task=fault_detection # fault_detection or classification

python run_main.py \
    --task $task \
    --comment "$task using $model" \
    --name "${task}_vsb" \
    --root_path ./dataset \
    --data_path VSBdata/vsb_images \
    --output_dir ./tensorboard \
    --records_file vsb_$task.xlsx \
    --model_name $model \
    --model_pretrain $model_pretrain \
    --epochs $epochs \
    --loss bce \
    --key_metric loss \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --d_model 128 \
    --dropout 0.1 \
    --patience 5 \
    --phase_level \
    # --weight_decay 1e-3 \
