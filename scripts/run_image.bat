@echo off

REM =========================
REM 进入项目目录（改成你的路径）
REM =========================
cd /d  E:\Graduate\projects\multimodal_vsb_20251208\research\code

REM =========================
REM 激活 conda 环境（改成你的路径和环境名）
REM =========================
call D:\Programming\python\conda\Scripts\activate.bat pd

REM =========================
REM 参数设置
REM =========================
set model=swin_transformer
set model_pretrain=microsoft/swinv2-tiny-patch4-window8-256
set batch_size=16
set epochs=10
set lr=1e-5
set itr=1
set task=fault_detection

REM =========================
REM 运行训练
REM =========================
python run_main.py ^
    --task %task% ^
    --comment "%task% using %model%" ^
    --name "%task%_vsb" ^
    --root_path ./dataset ^
    --data_path VSBdata/vsb_images ^
    --output_dir ./tensorboard ^
    --records_file vsb_%task%.xlsx ^
    --model_name %model% ^
    --model_pretrain %model_pretrain% ^
    --epochs %epochs% ^
    --loss bce ^
    --key_metric loss ^
    --seed 2025 ^
    --batch_size %batch_size% ^
    --lr %lr% ^
    --itr %itr% ^
    --d_model 128 ^
    --dropout 0.1 ^
    --patience 5 ^
    >> nohup.out 2>&1

echo Training finished.
