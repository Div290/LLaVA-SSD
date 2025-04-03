#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=$(python -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile)")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

export HF_HUB_DISABLE_SSL_VERIFY=1
echo $CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()"
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export PYTORCH_CUDA_GRAPH_CACHE_LAZY_FREE=1
export BNB_QUANTIZATION=True

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --bits 4 \
    --model_name_or_path /home/iitb/LLaVA/checkpoints \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct_80k.json \
    --image_folder /home/iitb/LLaVA/playground/datasets/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /home/iitb/LLaVA/checkpoints/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
