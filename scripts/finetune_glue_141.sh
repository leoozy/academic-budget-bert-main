aim=-bertbase--23000
model_path=/data1/zjl/output/24hb/pretraining_experiment-bertbase-baseline/pretraining_experiment-bertbase-baseline-/epoch1000000_step23025
model_name=/pretraining_experiment-bertbase-baseline-23000-bertbase
python run_glue.py \
  --model_name_or_path ${model_path} \
  --task_name cola \
  --max_seq_length 128 \
  --output_dir /data1/zjl/output/24hb/finetune/sst2/${model_name} \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 50 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50