max_steps=$1
aim=bert-mlm-augskip-${max_steps}-1e-3
deepspeed --include=localhost:0,1,2,3 --master_port 62000 run_pretraining.py \
  --model_type bert-mlm-augskip --tokenizer_name bert-base-uncased \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 128 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 24.0 \
  --early_exit_time_marker 24.0 \
  --dataset_path /data1/zjl/dataset/generated_samples_without_merge/ \
  --output_dir /data1/zjl/output/24hbert/${aim} \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name ${aim} \
  --project_name 24hb \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 180 \
  --early_stop_eval_loss 6 \
  --seed 42 \
  --fp16 \
  --max_steps ${max_steps} \
  --finetune_checkpoint_at_end
