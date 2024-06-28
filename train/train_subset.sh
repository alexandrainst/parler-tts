# accelerate launch ./training/run_parler_tts_training.py \
#     --model_name_or_path "./parler-tts-untrained-600M/parler-tts-untrained-600M" \
#     --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
#     --description_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
#     --prompt_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
#     --report_to "wandb" \
#     --overwrite_output_dir true \
#     --train_dataset_name "alexandrainst/nota-tts-small" \
#     --train_metadata_dataset_name "alexandrainst/nota-tts-small-tagged" \
#     --train_dataset_config_name "default" \
#     --train_split_name "train" \
#     --eval_dataset_name "alexandrainst/nota-tts-small" \
#     --eval_metadata_dataset_name "alexandrainst/nota-tts-small-tagged" \
#     --eval_dataset_config_name "default" \
#     --eval_split_name "train" \
#     --max_eval_samples 8 \
#     --per_device_eval_batch_size 8 \
#     --target_audio_column_name "audio_path" \
#     --description_column_name "description" \
#     --prompt_column_name "transcript" \
#     --max_duration_in_seconds 20.0 \
#     --min_duration_in_seconds 2.0 \
#     --max_text_length 400 \
#     --do_train true \
#     --num_train_epochs 10 \
#     --gradient_accumulation_steps 1 \
#     --gradient_checkpointing true \
#     --per_device_train_batch_size 1 \
#     --learning_rate 0.0001 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.99 \
#     --weight_decay 0.01 \
#     --lr_scheduler_type "constant_with_warmup" \
#     --warmup_steps 2 \
#     --logging_steps 2 \
#     --freeze_text_encoder true \
#     --audio_encoder_per_device_batch_size 4 \
#     --dtype "float16" \
#     --seed 456 \
#     --output_dir "/mnt/nota-NAS/output_dir_training/" \
#     --temporary_save_to_disk "/mnt/nota-NAS/audio_code_tmp/" \
#     --save_to_disk "/mnt/nota-NAS/tmp_dataset_audio/" \
#     --dataloader_num_workers 2 \
#     --predict_with_generate \
#     --include_inputs_for_metrics \
#     --group_by_length true \
#     --asr_model_name_or_path "jstoone/whisper-medium-da" \
#     --id_column_name "id" \
#     --token "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" \
#     --preprocessing_num_workers 8 \
#     --do_eval \
#     --preprocessing_only


# Lokale filer
accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "./parler-tts-untrained-600M/parler-tts-untrained-600M" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
    --prompt_tokenizer_name "RJuro/munin-neuralbeagle-7b" \
    --report_to "wandb" \
    --overwrite_output_dir true \
    --train_dataset_name "/home/alex-admin/nota_tts/data/final/dataset" \
    --train_metadata_dataset_name "data/final/dataset-tagged" \
    --train_dataset_config_name "default" \
    --train_split_name "train" \
    --eval_dataset_name "/home/alex-admin/nota_tts/data/final/dataset" \
    --eval_metadata_dataset_name "data/final/dataset-tagged" \
    --eval_dataset_config_name "default" \
    --eval_split_name "dev" \
    --max_eval_samples 8 \
    --per_device_eval_batch_size 8 \
    --target_audio_column_name "audio_path" \
    --description_column_name "description" \
    --prompt_column_name "transcript" \
    --max_duration_in_seconds 20.0 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 400 \
    --do_train true \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 1 \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 2 \
    --logging_steps 2 \
    --freeze_text_encoder true \
    --audio_encoder_per_device_batch_size 4 \
    --dtype "float16" \
    --seed 456 \
    --output_dir "/mnt/nota-NAS/output_dir_training/" \
    --temporary_save_to_disk "/mnt/nota-NAS/audio_code_tmp/" \
    --save_to_disk "/mnt/nota-NAS/tmp_dataset_audio/" \
    --dataloader_num_workers 2 \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --group_by_length true \
    --asr_model_name_or_path "jstoone/whisper-medium-da" \
    --id_column_name "id" \
    --token "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" \
    --preprocessing_num_workers 8 \
    --cache_dir "/home/alex-admin/.cache" \
    --do_eval \
    --preprocessing_only