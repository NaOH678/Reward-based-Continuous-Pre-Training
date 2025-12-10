export SENTRY_DSN=""
export WANDB_API_KEY="bb8507670acdf8e05b2059b640b4e1d268b96aae"
export WANDB_MODE=offline
LOG_RANK=0 bash train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/openthoughts-test \
    --model.config /mnt/shared-storage-user/shichaojian/Llama3.2-1B/snapshots/5d853ed7d16ac794afa8f5c9c7f59f4e9c950954/config.json \
    --model.tokenizer_path /mnt/shared-storage-user/shichaojian/Llama3.2-1B/snapshots/5d853ed7d16ac794afa8f5c9c7f59f4e9c950954 \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr 5e-4 \
    --lr_scheduler.warmup_steps 2048 \
    --lr_scheduler.decay_type cosine \
    --training.batch_size 1 \
    --training.seq_len 15700 \
    --training.context_len 4096 \
    --training.gradient_accumulation_steps 16 \
    --training.epochs 3 \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset parquet \
    --training.dataset ../../formalverification-shared/shichaojian/OpenThoughts/data \
    --training.trust_remote_code \
    --training.dataset_split train \
    --training.num_workers 15 \
    --training.prefetch_factor 2 \
    --training.seed 42 \
    --training.sample_level \
    --future_encoder.enable \
    --future_encoder.future_k '-1' \
    --future_encoder.summary_method attention \
    --future_encoder.loss_weight 0.1 \
    --checkpoint.load_step 0 \
    --checkpoint.keep_latest_k 2 \
    --metrics.log_freq 1 \