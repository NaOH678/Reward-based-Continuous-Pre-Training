# ../../formalverification-shared/shichaojian/OpenThoughts/data 
# FLEX_DEBUG=1 to debug flex attn 
#--training.dataset parquet \
#--training.data_files /mnt/shared-storage-user/shichaojian/Reward-based-Continuous-Pre-Training/omni-math/omni-math.parquet \
# or

# --training.dataset ../../formalverification-shared/shichaojian/domino-50B/
#  llama /mnt/shared-storage-user/shichaojian/Llama3.2-1B/snapshots/5d853ed7d16ac794afa8f5c9c7f59f4e9c950954
# olmo /mnt/shared-storage-user/shichaojian/OLMo-1B/dolma-tokenizer
# olmo /mnt/shared-storage-user/shichaojian/OLMo-1B/config.json

#ddp
# --training.data_parallel_replicate_degree 8 \
# --training.data_parallel_shard_degree 1 \


# varlen
# --training.varlen \

# batch level cu_seqlens
# --future_encoder.respect_doc_boundaries \
export SENTRY_DSN=""
export WANDB_API_KEY="bb8507670acdf8e05b2059b640b4e1d268b96aae"
export WANDB_MODE=offline
LOG_RANK=0 bash train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/dolmino-mi-olmo-docbordline-shuffle-k0 \
    --model.config  /mnt/shared-storage-user/shichaojian/OLMo-1B/config.json \
    --model.tokenizer_path  /mnt/shared-storage-user/shichaojian/OLMo-1B/dolma-tokenizer \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr 0.000074487 \
    --lr_scheduler.warmup_steps 0 \
    --lr_scheduler.decay_type linear \
    --training.batch_size 4 \
    --training.seq_len 4096 \
    --training.context_len 4096 \
    --training.gradient_accumulation_steps 4 \
    --training.epochs 1 \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset ../../formalverification-shared/shichaojian/domino-50B-shuffled/ \
    --training.trust_remote_code \
    --training.dataset_split train \
    --training.num_workers 15 \
    --training.prefetch_factor 2 \
    --training.seed 42 \
    --future_encoder.enable \
    --future_encoder.respect_doc_boundaries \
    --future_encoder.temperature 0.1 \
    --future_encoder.future_k 0 \
    --future_encoder.loss_weight 0.1 \
    --future_predictor.enable \
    --future_predictor.head_type mlp \
    --checkpoint.initial_load_path ../OLMo-1B/checkpoint_final/checkpoint/step-0 \
    --checkpoint.load_step 0 \
    --checkpoint.interval 1000 \
    --checkpoint.keep_latest_k 2 \
    --metrics.log_freq 1 \