accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "./parler-tts-untrained-600M/parler-tts-untrained-600M" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
    --prompt_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
    --report_to "wandb" \
    --overwrite_output_dir true \
    --load_from_disk "/home/alex-admin/nota_tts/data/final/dataset-split-train" \
    --train_dataset_config_name "default" \
    --eval_dataset_config_name "default" \
    --eval_split_name "dev" \
    --max_eval_samples 8 \
    --per_device_eval_batch_size 8 \
    --target_audio_column_name "audio_path" \
    --description_column_name "description" \
    --prompt_column_name "transcript" \
    --max_duration_in_seconds 30.0 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 700 \
    --do_train true \
    --num_train_epochs 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 3 \
    --learning_rate 0.000095 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 3000 \
    --logging_steps 50 \
    --freeze_text_encoder true \
    --audio_encoder_per_device_batch_size 5 \
    --dtype "float16" \
    --seed 456 \
    --output_dir "/mnt/nota-NAS/output_dir_training/" \
    --temporary_save_to_disk "/mnt/nota-NAS/audio_code_tmp/" \
    --save_to_disk "/mnt/nota-NAS/tmp_dataset_audio/" \
    --dataloader_num_workers 8 \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --group_by_length true \
    --asr_model_name_or_path "jstoone/whisper-medium-da" \
    --id_column_name "id" \
    --preprocessing_num_workers 1 \
    --cache_dir "/home/alex-admin/.cache" \
    --token "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" \
    --trust_remote_code true \
    --cache_dir "/home/alex-admin/.cache" \
    --resume_from_checkpoint "/mnt/nota-NAS/output_dir_training/checkpoint-160000-epoch-7" \
    --save_steps 10000 
    # --preprocessing_only
    # --do_eval \
    


# # --dtype "float16"
# # google/flan-t5-base
#     --description_tokenizer_name "google/flan-t5-base" \
#     --prompt_tokenizer_name "google/flan-t5-base" \
#     # --description_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
#     # --prompt_tokenizer_name "RJuro/munin-neuralbeagle-7b" \