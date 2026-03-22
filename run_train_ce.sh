# --checkpoint.initial_load_path ../OLMo-1B/checkpoint_final/checkpoint/step-0 \
# /mnt/shared-storage-user/shichaojian/OLMo-1B/checkpoint_nested/checkpoint/step-0

#     --training.varlen \
export SENTRY_DSN=""
export WANDB_API_KEY="bb8507670acdf8e05b2059b640b4e1d268b96aae"
export WANDB_MODE=offline
LOG_RANK=0 bash train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/dolmino-ce-olmo-baseline-initial_test \
    --model.config  /mnt/shared-storage-user/shichaojian/OLMo-1B/config.json \
    --model.tokenizer_path  /mnt/shared-storage-user/shichaojian/OLMo-1B/dolma-tokenizer \
    --optimizer.name AdamW \
    --optimizer.eps 1e-8 \
    --optimizer.lr 0.000074487 \
    --lr_scheduler.warmup_steps 0 \
    --lr_scheduler.decay_type linear \
    --training.batch_size 4 \
    --training.seq_len 4096 \
    --training.context_len 4096 \
    --training.gradient_accumulation_steps 16 \
    --training.epochs 1 \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset ../../formalverification-shared/shichaojian/domino-50B-fullshuffled/ \
    --training.trust_remote_code \
    --training.dataset_split train \
    --training.num_workers 15 \
    --training.prefetch_factor 2 \
    --training.seed 42 \
    --checkpoint.initial_load_path ../OLMo-1B/fla-future_predictor/checkpoint/step-0 \
    --checkpoint.load_step 0 \
    --checkpoint.interval 100 \
    --checkpoint.keep_latest_k 2 \
    --metrics.log_freq 1 \