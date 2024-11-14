
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 8B, 70B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
SP=${13}                        # 是否使用序列并行: true, false
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否优先使用Flash Attention: true, false
SFT=${16}                       # 是否执行微调训练: true, false
AC=${17}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${18}         # 是否启用Offload optimizer: false, static, auto
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
VALID_DATASET_PATH=${21}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${23}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${24}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${25}           # 训练输出日志文件路径

# torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 48220 pretrain_llama.py --save ./result/output_mcore_llama3_2_1B/checkpoint/finetune-mcore-llama3-1-1B-lr-1e-5-minlr-1e-6-bs-32-gbs-32-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-false-ti-10000-wi-100 --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --init-method-std 0.008 --attention-dropout 0.0 --hidden-dropout 0.0 --lr-decay-iters 9900 --lr-warmup-iters 100 --train-iters 10000 --micro-batch-size 32 --global-batch-size 32 --num-layers 16 --hidden-size 2048 --num-attention-heads 32 --ffn-hidden-size 8192 --seq-length 128 --max-position-embeddings 131072 --max-padding-length 128 --log-interval 1 --log-throughput --eval-interval 10000 --eval-iters 10 --save-interval 100000 --tensorboard-queue-size 1 --tensorboard-dir ./result/output_mcore_llama3_2_1B/tensorboard/finetune-mcore-llama3-1-1B-lr-1e-5-minlr-1e-6-bs-32-gbs-32-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-false-ti-10000-wi-100_2024.10.31-13.32.05 --log-timers-to-tensorboard --log-batch-size-to-tensorboard --log-validation-ppl-to-tensorboard --log-memory-to-tensorboard --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --no-load-optim --no-load-rng --num-workers 8 --extra-vocab-size 256 --patch-tokenizer-type LLama3Tokenizer --swiglu --normalization RMSNorm --norm-epsilon 1e-05 --use-rotary-position-embeddings --position-embedding-type rope --untie-embeddings-and-output-weights --disable-bias-linear --rotary-base 500000 --no-save-optim --train-data-path /home/scc/datasets/alpaca_zh-llama3-train.json --valid-data-path /home/scc/datasets/alpaca_zh-llama3-valid.json --dataloader-type cyclic --dataset LLama-SFT-Raw --bf16 --load /home/scc/models/Llama-3.2-1B-Instruct/mcore-tp1-pp1 --transformer-impl transformer_engine --recompute-activations --use-distributed-optimizer --group-query-attention --num-query-groups 8 --optimizer hybridadam --optimizer-offload-policy static --optimizer-offload-fraction 1.0 --overlap-grad-reduce --overlap-param-gather --eod-mask-loss --train-mode finetune

# 4090 24G单卡 PAI源码 1B 只能开启offload跑，否则会OOM，其余不行
MP_DATASET_TYPE="raw" sh run_mcore_llama3_1.sh \
    dsw \
    1B \
    32 \
    32 \
    1e-5 \
    1e-6 \
    128 \
    128 \
    bf16 \
    1 \
    1 \
    1 \
    false \
    false \
    false \
    true \
    sel \
    static \
    100000 \
    /home/scc/datasets/alpaca_zh-llama3-train.json \
    /home/scc/datasets/alpaca_zh-llama3-valid.json \
    /home/scc/models/Llama-3.2-1B-Instruct/mcore-tp1-pp1 \
    10000 \
    100 \
    ./result/output_mcore_llama3_2_1B



# 4090 24G单卡尝试8B OOM 开启offload也无用
sh run_mcore_llama3_1.sh \
    dsw \
    8B \ 
    1 \ 
    1 \ 
    1e-5 \
    1e-6 \
    128 \
    128 \
    bf16 \
    1 \
    1 \
    1 \
    false \
    true \
    true \
    true \
    offload \
    auto \
    100000 \
    /home/scc/datasets/alpaca_zh-llama3-train.json \
    /home/scc/datasets/alpaca_zh-llama3-valid.json \
    /home/scc/models/Meta-Llama-3.1-8B-Instruct/mcore-tp1-pp1 \
    10000 \
    100 \
    ./result/output_mcore_llama3_1 \

# convertor

MODEL_SIZE=$1                 # 模型参数：8B/70B
SOURCE_CKPT_PATH=$2           # 源checkpoint路径
TARGET_CKPT_PATH=$3           # 目标checkpoint路径
TP=$4                         # 模型并行度
PP=$5                         # 流水并行度
mg2hf=$6                      # 是否执行mcore2hf转换
CHECK=$7                      # 测试转换前后模型逐层输出是否一致
CHECK_ONLY=$8                 # 仅检测模型输出，不进行转换
PR=$9                         # 精度设置，fp16/bf16/fp32     
HF_CKPT_PATH=$10              # HF的CKPT的路径【可选，mg2hf=true时必须提供】

bash hf2mcore_convertor_llama3_1.sh \
1B \
/home/scc/models/Llama-3.2-1B-Instruct  \
/home/scc/models/Llama-3.2-1B-Instruct/mcore-tp1-pp1  \
1  \
1  \
false \
true \
false \
bf16